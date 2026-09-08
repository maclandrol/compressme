"""Small regressions for the audit's package-independent tensor-byte check."""
import unittest
import torch
from probe import tensor_byte_comparison


class TensorByteComparisonTests(unittest.TestCase):
    def test_signed_zero_is_distinct(self):
        for dtype in (torch.float32, torch.float64):
            left=torch.tensor([0.0],dtype=dtype)
            right=torch.tensor([-0.0],dtype=dtype)
            self.assertTrue(torch.equal(left,right))
            result=tensor_byte_comparison(left,right)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['failures'][0]['different_bytes'],1)

    def test_every_dtype_and_empty_leaf_is_counted(self):
        value={'float':torch.tensor([0.0]),'int':torch.tensor([2],dtype=torch.int64),
               'bool':torch.tensor([True]),'empty':torch.empty(0)}
        result=tensor_byte_comparison(value,{k:v.clone() for k,v in value.items()})
        self.assertTrue(result['accepted'])
        self.assertEqual(result['tensor_leaves'],4)
        self.assertEqual(result['bytes_checked'],13)
        self.assertEqual(result['tensor_leaves_by_dtype'],
                         {'torch.float32':2,'torch.int64':1,'torch.bool':1})

    def test_noncontiguous_logical_values(self):
        value=torch.arange(12,dtype=torch.float32).reshape(3,4)[:,::2]
        self.assertFalse(value.is_contiguous())
        self.assertTrue(tensor_byte_comparison(value,value.clone())['accepted'])

    def test_shape_dtype_and_extra_leaves_are_rejected(self):
        self.assertFalse(tensor_byte_comparison(torch.tensor([1]),torch.tensor([1.0]))['accepted'])
        self.assertFalse(tensor_byte_comparison(torch.tensor([1]),torch.tensor([[1]]))['accepted'])
        self.assertFalse(tensor_byte_comparison({'a':torch.tensor([1])},
                                               {'a':torch.tensor([1]),'b':torch.tensor([2])})['accepted'])

    def test_score_reconstruction_preserves_finite_zero_signs(self):
        values=torch.tensor([-0.0,0.0,float('-inf')])
        finite=torch.isfinite(values)
        reconstructed=torch.where(finite,values,torch.zeros_like(values))
        self.assertTrue(torch.signbit(reconstructed[0]))
        self.assertFalse(torch.signbit(reconstructed[1]))
        self.assertTrue(tensor_byte_comparison(reconstructed[:2],values[:2])['accepted'])


if __name__=='__main__':unittest.main()
