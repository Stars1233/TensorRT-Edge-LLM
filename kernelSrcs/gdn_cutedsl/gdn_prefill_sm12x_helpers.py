# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 FlashInfer team.
#
# Adapted from FlashInfer commit d020372b068f335e2fe427372e134977a2235c49
# for TensorRT Edge-LLM Blackwell GeForce GDN prefill.

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass._mlir.dialects.cute as _cute_ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T
from cutlass.utils.tensormap_manager import TensorMapManager, TensorMapUpdateMode

TENSOR_MAP_DESCRIPTOR_BYTES = 128


def round_down(a: int, b: int) -> int:
    return (a // b) * b


@cute.jit
def select_tensor_10(t: cute.Tensor) -> cute.Tensor:
    """Swap the first two modes of a tensor without moving data."""
    return cute.make_tensor(
        t.iterator.align(t.iterator.max_alignment),
        cute.make_layout(
            (t.layout.shape[1], t.layout.shape[0]) + t.layout.shape[2:],
            stride=(t.layout.stride[1], t.layout.stride[0]) + t.layout.stride[2:],
        ),
    )


@cute.jit
def smid():
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [],
            "mov.u32 $0, %smid;",
            "=r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def tensormap_replace_global_dim_1(
    tensormap_ptr: cute.Pointer,
    new_val: cutlass.Int32,
):
    ptr_i64 = tensormap_ptr.toint().ir_value()
    llvm.inline_asm(
        None,
        [ptr_i64, new_val.ir_value()],
        "tensormap.replace.tile.global_dim.global.b1024.b32 [$0], 1, $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


class SM80:
    @staticmethod
    @cute.jit
    def convert_c_layout_to_a_layout(c_layout, tiled_mma):
        c_frag_atom_size = cute.size(c_layout, mode=[0])
        a_frag_atom_size = cute.size(tiled_mma.tv_layout_A, mode=[1])
        ratio = a_frag_atom_size // c_frag_atom_size
        if cutlass.const_expr(ratio == 1):
            return c_layout

        divided = cute.logical_divide(c_layout, (None, None, ratio))
        frag_layout = cute.flatten(
            cute.make_layout(
                (divided.shape[0], divided.shape[2][0]),
                stride=(divided.stride[0], divided.stride[2][0]),
            )
        )
        return cute.make_layout(
            (frag_layout.shape, divided.shape[1], divided.shape[2][1]),
            stride=(
                frag_layout.stride,
                divided.stride[1],
                divided.stride[2][1],
            ),
        )

    @staticmethod
    @cute.jit
    def make_acc_into_op(acc: cute.Tensor, tiled_mma, dtype) -> cute.Tensor:
        operand = cute.make_fragment_like(
            SM80.convert_c_layout_to_a_layout(acc.layout, tiled_mma),
            dtype,
        )
        operand_as_acc = cute.make_tensor(operand.iterator, acc.layout)
        operand_as_acc.store(acc.load().to(dtype))
        return operand


class CollectiveStoreTma:
    def __init__(self, blk_q: int, d: int):
        self.BLK_Q = blk_q
        self.D = d

    @cute.jit
    def tail_tensormap_gmem_ptr(self, g_tensormaps: cute.Tensor):
        manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, TENSOR_MAP_DESCRIPTOR_BYTES
        )
        return manager.get_tensormap_ptr(
            g_tensormaps.iterator
            + smid() * cutlass.Int32(TENSOR_MAP_DESCRIPTOR_BYTES)
        )

    @cute.jit
    def tail_tensormap_generic_ptr(self, g_tensormaps: cute.Tensor):
        manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, TENSOR_MAP_DESCRIPTOR_BYTES
        )
        return manager.get_tensormap_ptr(
            g_tensormaps.iterator
            + smid() * cutlass.Int32(TENSOR_MAP_DESCRIPTOR_BYTES),
            address_space=_cute_ir.AddressSpace.generic,
        )

    @cute.jit
    def can_process(
        self,
        work_desc,
        blk: cutlass.Int32,
        num_blocks: cutlass.Int32,
    ):
        can_process = blk < num_blocks - cutlass.Int32(1)
        if work_desc.seq_len % cutlass.Int32(self.BLK_Q) == cutlass.Int32(0):
            can_process = True
        return can_process

    @cute.jit
    def create_tensormap_for_tail(
        self,
        tma_atom_o: cute.CopyAtom,
        g_tensormaps: cute.Tensor,
        work_desc,
    ):
        tail_ptr = self.tail_tensormap_gmem_ptr(g_tensormaps)
        with cute.arch.elect_one():
            cpasync.copy_tensormap(tma_atom_o, tail_ptr)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            tensormap_replace_global_dim_1(
                tail_ptr,
                work_desc.tok_offset + work_desc.seq_len,
            )
        cute.arch.sync_warp()
        cpasync.fence_tma_desc_release()

    @cute.jit
    def partition_sd(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        stage_idx: cutlass.Int32,
    ):
        mO = cute.domain_offset(
            (cutlass.Int32(0), work_desc.tok_offset + blk * cutlass.Int32(self.BLK_Q)),
            tma_tensor_o[None, None, o_head_idx],
        )
        gO = cute.zipped_divide(mO, (self.D, self.BLK_Q))[
            ((None, None), (cutlass.Int32(0), cutlass.Int32(0)))
        ]
        sO_pipe = sO[None, None, stage_idx]
        return cpasync.tma_partition(
            tma_atom_o,
            0,
            cute.make_layout(1),
            cute.group_modes(sO_pipe, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

    @cute.jit
    def issue_store(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        stage_idx: cutlass.Int32,
    ):
        cute.arch.fence_view_async_shared()
        tOsO, tOgO = self.partition_sd(
            sO, tma_atom_o, tma_tensor_o, work_desc, o_head_idx, blk, stage_idx
        )
        cute.copy(tma_atom_o, tOsO, tOgO)
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def issue_tail_store(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_tensormaps: cute.Tensor,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        stage_idx: cutlass.Int32,
    ):
        cute.arch.fence_view_async_shared()
        tOsO, tOgO = self.partition_sd(
            sO, tma_atom_o, tma_tensor_o, work_desc, o_head_idx, blk, stage_idx
        )
        tail_gmem_ptr = self.tail_tensormap_gmem_ptr(g_tensormaps)
        tail_generic_ptr = self.tail_tensormap_generic_ptr(g_tensormaps)
        cpasync.fence_tma_desc_acquire(tail_gmem_ptr)
        cute.copy(
            tma_atom_o,
            tOsO,
            tOgO,
            tma_desc_ptr=tail_generic_ptr,
        )
        cute.arch.cp_async_bulk_commit_group()

    @cute.jit
    def step(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_tensormaps: cute.Tensor,
        o_pipeline,
        o_consumer_state,
        work_desc,
        o_head_idx: cutlass.Int32,
        blk: cutlass.Int32,
        num_blocks: cutlass.Int32,
    ):
        if blk == cutlass.Int32(0) and not self.can_process(
            work_desc, num_blocks - cutlass.Int32(1), num_blocks
        ):
            self.create_tensormap_for_tail(tma_atom_o, g_tensormaps, work_desc)

        o_pipeline.consumer_wait(o_consumer_state)
        if self.can_process(work_desc, blk, num_blocks):
            self.issue_store(
                sO,
                tma_atom_o,
                tma_tensor_o,
                work_desc,
                o_head_idx,
                blk,
                o_consumer_state.index,
            )
        else:
            self.issue_tail_store(
                sO,
                tma_atom_o,
                tma_tensor_o,
                g_tensormaps,
                work_desc,
                o_head_idx,
                blk,
                o_consumer_state.index,
            )
        cute.arch.cp_async_bulk_wait_group(0)
        o_pipeline.consumer_release(o_consumer_state)
        o_consumer_state.advance()
        return o_consumer_state

    @cute.jit
    def run(
        self,
        sO: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        tma_tensor_o: cute.Tensor,
        g_tensormaps: cute.Tensor,
        o_pipeline,
        num_blocks: cutlass.Int32,
        work_desc,
        o_stage: int,
        num_q_heads: cutlass.Int32,
        num_v_heads: cutlass.Int32,
    ):
        o_head_idx = work_desc.o_head_idx(num_q_heads, num_v_heads)
        o_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, o_stage
        )
        for blk in cutlass.range(num_blocks, unroll=1):
            o_consumer_state = self.step(
                sO,
                tma_atom_o,
                tma_tensor_o,
                g_tensormaps,
                o_pipeline,
                o_consumer_state,
                work_desc,
                o_head_idx,
                blk,
                num_blocks,
            )
