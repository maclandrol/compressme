import hashlib
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch
import zstandard as zstd
import packing_files as stream

# The installed existing API is read-only and remains unchanged by this experiment.
from compressme.packing import pack_bytes, unpack_bytes

class PackingFiles(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
    def tearDown(self):self.temp.cleanup()
    def paths(self,data):
        source=self.root/'source';source.write_bytes(data)
        return source,self.root/'packed',self.root/'output'
    def test_cross_compatibility(self):
        for group in (1,2,4,8):
            for n in (0,1,7,64,65,2049):
                data=bytes((i*37)%256 for i in range(n))
                source,packed,output=self.paths(data)
                report=stream.pack_file(source,packed,group_size=group,block_size=64,overwrite=True)
                self.assertEqual(unpack_bytes(packed.read_bytes()),data)
                self.assertEqual(report['sha256'],hashlib.sha256(data).hexdigest())
                packed.write_bytes(pack_bytes(data,group_size=group,block_size=64))
                stream.unpack_file(packed,output,overwrite=True)
                self.assertEqual(output.read_bytes(),data)
    def test_multiblock_realistic_arbitrary_bytes(self):
        data=os.urandom(2*(1<<20)+19)
        source,packed,output=self.paths(data)
        stream.pack_file(source,packed)
        stream.unpack_file(packed,output,max_output_bytes=len(data))
        self.assertEqual(output.read_bytes(),data)
    def test_small_file_truncation_every_boundary(self):
        data=pack_bytes(b'hello world'*9)
        _,packed,output=self.paths(b'')
        for n in range(len(data)):
            packed.write_bytes(data[:n])
            with self.assertRaises((ValueError,struct.error)):stream.unpack_file(packed,output)
            self.assertFalse(output.exists())
            self.assertFalse(list(self.root.glob('.output.*.tmp')))
    def test_trailing_bytes_frames_and_skippable_frames(self):
        _,packed,output=self.paths(b'')
        for tail in (b'\0',b'extra',zstd.ZstdCompressor().compress(b''),b'\x50\x2a\x4d\x18\x00\x00\x00\x00'):
            packed.write_bytes(pack_bytes(b'abc')+tail)
            with self.assertRaises(ValueError):stream.unpack_file(packed,output)
            self.assertFalse(output.exists())
    def test_corruption_header_hash_and_frame_checksum(self):
        encoded=pack_bytes(b'abcdefgh'*500)
        _,packed,output=self.paths(b'')
        for index in (0,8,9,10,11,12,23,25,len(encoded)-1):
            mutated=bytearray(encoded);mutated[index]^=1;packed.write_bytes(mutated)
            with self.assertRaises(ValueError):stream.unpack_file(packed,output)
            self.assertFalse(output.exists())
    def test_size_cap_before_output_creation(self):
        _,packed,output=self.paths(b'');packed.write_bytes(pack_bytes(b'x'*2048))
        with self.assertRaisesRegex(ValueError,'max_output_bytes'):stream.unpack_file(packed,output,max_output_bytes=2047)
        self.assertFalse(output.exists())
        for bad in (None,-1,True,1.5):
            with self.assertRaises(ValueError):stream.unpack_file(packed,output,max_output_bytes=bad)
    def test_window_cap(self):
        _,packed,output=self.paths(b'');packed.write_bytes(pack_bytes(b'x'*2048))
        with self.assertRaisesRegex(ValueError,'max_window_bytes'):stream.unpack_file(packed,output,max_window_bytes=1024)
        self.assertFalse(output.exists())
    def test_declared_and_frame_size_disagree(self):
        encoded=bytearray(pack_bytes(b'abc'));struct.pack_into('>Q',encoded,16,4)
        _,packed,output=self.paths(b'');packed.write_bytes(encoded)
        with self.assertRaisesRegex(ValueError,'sizes disagree'):stream.unpack_file(packed,output)
    def test_unknown_content_size_refused(self):
        data=b'abc';encoded=stream._HEADER.pack(stream.MAGIC,1,1,1,1,64,3,hashlib.sha256(data).digest())
        encoded+=zstd.ZstdCompressor(write_content_size=False).compress(data)
        _,packed,output=self.paths(b'');packed.write_bytes(encoded)
        with self.assertRaisesRegex(ValueError,'sizes disagree'):stream.unpack_file(packed,output)
    def test_no_overwrite_and_explicit_overwrite(self):
        source,packed,output=self.paths(b'abc');packed.write_bytes(b'old')
        with self.assertRaises(FileExistsError):stream.pack_file(source,packed)
        self.assertEqual(packed.read_bytes(),b'old')
        stream.pack_file(source,packed,overwrite=True);output.write_bytes(b'old')
        with self.assertRaises(FileExistsError):stream.unpack_file(packed,output)
        stream.unpack_file(packed,output,overwrite=True);self.assertEqual(output.read_bytes(),b'abc')
    def test_corrupt_input_preserves_existing_destination(self):
        source,packed,output=self.paths(b'abc');packed.write_bytes(pack_bytes(b'abc')[:-1]);output.write_bytes(b'keep')
        with self.assertRaises(ValueError):stream.unpack_file(packed,output,overwrite=True)
        self.assertEqual(output.read_bytes(),b'keep')
    def test_atomic_noclobber_race(self):
        source,packed,output=self.paths(b'abc');real_link=os.link
        def link(src,dst):Path(dst).write_bytes(b'racing target');return real_link(src,dst)
        with patch.object(stream.os,'link',link):
            with self.assertRaises(FileExistsError):stream.pack_file(source,packed)
        self.assertEqual(packed.read_bytes(),b'racing target');self.assertFalse(list(self.root.glob('.packed.*.tmp')))
    def test_same_file_and_hardlink_refused(self):
        source,packed,output=self.paths(b'abc');os.link(source,packed)
        for target in (source,packed):
            with self.assertRaises(ValueError):stream.pack_file(source,target,overwrite=True)
        self.assertEqual(source.read_bytes(),b'abc')
    def test_failing_codec_cleans_temporary_and_preserves_target(self):
        source,packed,output=self.paths(b'abc');packed.write_bytes(b'keep')
        with patch.object(stream,'_shuffle_block',side_effect=RuntimeError('planned')):
            with self.assertRaisesRegex(RuntimeError,'planned'):stream.pack_file(source,packed,overwrite=True)
        self.assertEqual(packed.read_bytes(),b'keep');self.assertFalse(list(self.root.glob('.packed.*.tmp')))
    def test_layout_validation(self):
        source,packed,_=self.paths(b'abc')
        for kwargs in ({'group_size':3},{'block_size':3},{'block_size':(16<<20)+4},{'level':True},{'level':23}):
            with self.assertRaises(ValueError):stream.pack_file(source,packed,**kwargs)
    def test_plain_and_rle_frame_structure(self):
        _,packed,output=self.paths(b'')
        for data,kind,body in ((b'abc',0,b'abc'),(b'x'*8,1,b'x')):
            frame=b'\x28\xb5\x2f\xfd'+bytes((0x20,len(data)))
            frame+=(1|(kind<<1)|(len(data)<<3)).to_bytes(3,'little')+body
            encoded=stream._HEADER.pack(stream.MAGIC,1,1,1,1,64,len(data),hashlib.sha256(data).digest())+frame
            packed.write_bytes(encoded);stream.unpack_file(packed,output,overwrite=True)
            self.assertEqual(output.read_bytes(),data)
    def test_reserved_and_oversized_block_rejected(self):
        _,packed,output=self.paths(b'')
        for header in (1|(3<<1),1|(131073<<3)):
            frame=b'\x28\xb5\x2f\xfd'+bytes((0x20,3))+header.to_bytes(3,'little')+b'abc'
            encoded=stream._HEADER.pack(stream.MAGIC,1,1,1,1,64,3,hashlib.sha256(b'abc').digest())+frame
            packed.write_bytes(encoded)
            with self.assertRaisesRegex(ValueError,'block header'):stream.unpack_file(packed,output)
            self.assertFalse(output.exists())
    def test_observed_input_mutation_rejected_transactionally(self):
        source,packed,output=self.paths(b'abc');real_token=stream._file_token;calls=[0]
        def token(handle):
            value=real_token(handle);calls[0]+=1
            return value if calls[0]==1 else value[:-1]+(value[-1]+1,)
        with patch.object(stream,'_file_token',token):
            with self.assertRaisesRegex(ValueError,'Source changed'):stream.pack_file(source,packed)
        self.assertFalse(packed.exists());self.assertFalse(list(self.root.glob('.packed.*.tmp')))
    def test_never_shuffles_more_than_one_block(self):
        data=os.urandom(65539);source,packed,output=self.paths(data);sizes=[];original=stream._shuffle_block
        def shuffle(raw,*args,**kwargs):sizes.append(len(raw));return original(raw,*args,**kwargs)
        with patch.object(stream,'_shuffle_block',shuffle):
            stream.pack_file(source,packed,block_size=4096)
            stream.unpack_file(packed,output)
        self.assertEqual(output.read_bytes(),data);self.assertLessEqual(max(sizes),4096)
    def test_decoder_limit_even_if_preinspection_is_bypassed(self):
        _,packed,output=self.paths(b'');packed.write_bytes(pack_bytes(b'x'*2048))
        with patch.object(stream,'_inspect_frame',return_value=2048):
            with self.assertRaisesRegex(ValueError,'Zstandard payload'):stream.unpack_file(packed,output,max_window_bytes=1024)
        self.assertFalse(output.exists())
    def test_runtime_window_unit_is_enforced(self):
        self.assertIn(stream._window_unit_bytes(zstd),(1,1024))
    @unittest.skipUnless(hasattr(os,'mkfifo') and hasattr(os,'O_NONBLOCK'),'requires POSIX FIFO')
    def test_fifo_source_refused_without_waiting_for_writer(self):
        fifo=self.root/'fifo';os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError,'regular file'):stream.pack_file(fifo,self.root/'output')
        self.assertFalse((self.root/'output').exists())
    def test_explicit_commit_point_if_temp_cleanup_fails(self):
        source,packed,_=self.paths(b'abc');original=Path.unlink
        def unlink(path,*args,**kwargs):
            if path.name.startswith('.packed.') and path.suffix=='.tmp':raise PermissionError('planned cleanup')
            return original(path,*args,**kwargs)
        with patch.object(Path,'unlink',unlink):
            with self.assertRaisesRegex(RuntimeError,'destination was committed'):stream.pack_file(source,packed)
        self.assertEqual(unpack_bytes(packed.read_bytes()),b'abc')
        for path in self.root.glob('.packed.*.tmp'):path.unlink()
    def test_empty_frame_checksum_and_extra_data_are_checked(self):
        encoded=pack_bytes(b'');_,packed,output=self.paths(b'')
        for payload in (encoded+b'extra',encoded[:-1]+bytes([encoded[-1]^1])):
            packed.write_bytes(payload)
            with self.assertRaises(ValueError):stream.unpack_file(packed,output)
            self.assertFalse(output.exists())
    def test_symbolic_float_payload_bits(self):
        data=bytes.fromhex('00000080000000000100c07f0100807f000080ff')+b'x'
        source,packed,output=self.paths(data);stream.pack_file(source,packed);stream.unpack_file(packed,output)
        self.assertEqual(output.read_bytes(),data)

if __name__=='__main__':unittest.main()
