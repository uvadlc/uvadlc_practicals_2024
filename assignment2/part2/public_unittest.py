import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from gpt import CausalSelfAttention  # Replace with the actual import path


class MockConfig:
    def __init__(self, n_embd=64, n_head=8, attn_pdrop=0, resid_pdrop=0.1, block_size=128, use_flash_attn=False, abs_emb=False):
        self.n_embd = n_embd
        self.n_head = n_head
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.block_size = block_size
        self.use_flash_attn = use_flash_attn
        self.abs_emb = abs_emb


class TestCausalSelfAttention(unittest.TestCase):
    def test_output_shape(self):
        config = MockConfig()
        module = CausalSelfAttention(config)
        B, T, C = 2, 10, config.n_embd
        x = torch.randn(B, T, C)
        output = module(x)
        self.assertEqual(output.shape, (B, T, C))

    def test_causal_masking(self):
        config = MockConfig()
        module = CausalSelfAttention(config, debug=True)
        B, T, C = 1, 5, config.n_embd
        x = torch.randn(B, T, C)
        outputs = module(x)
        att_probs = outputs['att_probs']
        for t in range(T):
            self.assertTrue(torch.allclose(att_probs[0, :, t, t+1:], torch.zeros_like(att_probs[0, :, t, t+1:])))

    def test_apply_rotary_emb(self):
        config = MockConfig()
        module = CausalSelfAttention(config)
        B, T, n_head, dim = 1, 5, config.n_head, config.n_embd // config.n_head
        xq = torch.randn(B, n_head, T, dim)
        xk = torch.randn(B, n_head, T, dim)
        xq_rot, xk_rot = module.apply_rotary_emb(xq, xk, T)
        self.assertEqual(xq_rot.shape, xq.shape)
        self.assertEqual(xk_rot.shape, xk.shape)
        self.assertTrue(torch.allclose(xq.norm(dim=-1), xq_rot.norm(dim=-1), atol=1e-5))
        self.assertTrue(torch.allclose(xk.norm(dim=-1), xk_rot.norm(dim=-1), atol=1e-5))


    def test_gradient_flow(self):
        config = MockConfig()
        module = CausalSelfAttention(config)
        B, T, C = 2, 10, config.n_embd
        x = torch.randn(B, T, C, requires_grad=True)
        output = module(x)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_attention_component_shapes_and_softmax(self):
        config = MockConfig()
        module = CausalSelfAttention(config, debug=True)
        B, T, C = 2, 5, config.n_embd
        x = torch.randn(B, T, C)
        outputs = module(x)
        att_probs, q, k, v = outputs["att_probs"], outputs["q"], outputs["k"], outputs["v"]

        # Check shapes
        self.assertEqual(att_probs.shape, (B, config.n_head, T, T), "Attention shape mismatch")
        self.assertEqual(q.shape, (B, config.n_head, T, C // config.n_head), "Query shape mismatch")
        self.assertEqual(k.shape, (B, config.n_head, T, C // config.n_head), "Key shape mismatch")
        self.assertEqual(v.shape, (B, config.n_head, T, C // config.n_head), "Value shape mismatch")

        # Check that attention weights sum to 1 after softmax
        self.assertTrue(torch.allclose(att_probs.sum(dim=-1), torch.ones_like(att_probs.sum(dim=-1)), atol=1e-5),
                        "Attention probabilities do not sum to 1")

if __name__ == '__main__':
    unittest.main()
