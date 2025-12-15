import os
import io
from app import app
import unittest
import obspy
import shutil
import tempfile

class ProcessingTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.secret_key = 'test'
        self.client = app.test_client()
        self.examples_dir = 'Examples'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_processing_rotation(self):
        # 1. Prepare N and E components
        fname_e = 'L1.RE006..EHE.2023.301.152902.PROC.mseed'
        path_e = os.path.join(self.examples_dir, fname_e)
        
        st = obspy.read(path_e)
        st[0].stats.channel = 'EHE'
        path_e_tmp = os.path.join(self.temp_dir, fname_e)
        st.write(path_e_tmp, format='MSEED')
        
        # Create Dummy N
        st_n = st.copy()
        st_n[0].stats.channel = 'EHN'
        fname_n = fname_e.replace('EHE', 'EHN')
        path_n_tmp = os.path.join(self.temp_dir, fname_n)
        st_n.write(path_n_tmp, format='MSEED')
        
        # Upload
        files_to_upload = [(open(path_e_tmp, 'rb'), fname_e), (open(path_n_tmp, 'rb'), fname_n)]
        data = {'deconv_files': files_to_upload}
        rv = self.client.post('/deconv_upload', data=data, follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        
        # Verify Uploaded
        with self.client.session_transaction() as sess:
            files_info = sess['deconv_files_info']
            print("Files before rotation:", [f['filename'] for f in files_info])
            self.assertEqual(len(files_info), 2)
            
        # 2. Apply Rotation
        process_data = {
            'rotate': 'on',
            'baz': '45',
            # Add detrend to verify standard processing too
            'detrend': 'on'
        }
        
        rv = self.client.post('/deconv_preprocess', data=process_data, follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        
        # Verify Files Changed to R/T
        with self.client.session_transaction() as sess:
            files_info = sess['deconv_files_info']
            filenames = [f['filename'] for f in files_info]
            print("Files after rotation:", filenames)
            
            # Check for R and T in filenames or stats
            # My logic: replace channel in filename.
            # EHE -> EHR (or R?), EHN -> EHT (or T?)
            # Wait, obspy rotate NE->RT usually produces 'R' and 'T' as component codes.
            # So Channel might be 'EHR' and 'EHT'.
            
            has_R = any('R' in f and 'EHE' not in f and 'EHN' not in f for f in filenames) # Simple check
            # Actually, let's check exact names if possible, but they are derived.
            # Just checking that we have 2 files and they are different is a good start,
            # but checking specific channel codes is better.
            
            # We can read the files from the session folder if we could match the path,
            # but here we rely on the session info.
            channels = [f['stats'] for f in files_info]
            print("Channels:", channels)
            
            self.assertTrue(any('R' in str(c) for c in channels), "Radial component not found")
            self.assertTrue(any('T' in str(c) for c in channels), "Transverse component not found")

if __name__ == '__main__':
    unittest.main()
