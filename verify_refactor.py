import os
import io
from app import app
import unittest
import obspy
import shutil
import tempfile

class RefactorTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.secret_key = 'test'
        self.client = app.test_client()
        self.examples_dir = 'Examples'
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_unified_workflow(self):
        # 1. Prepare Multiple Files
        fname_1 = 'L1.RE006..EHE.2023.301.152902.PROC.mseed'
        fname_2 = 'L1.RE008..EHE.2023.301.152902.PROC.mseed'
        
        path_1 = os.path.join(self.examples_dir, fname_1)
        path_2 = os.path.join(self.examples_dir, fname_2)

        # Mock opening files
        # Flask test client for multiple files needs list of (file, filename) with SAME KEY
        files_to_upload = [
            (open(path_1, 'rb'), fname_1),
            (open(path_2, 'rb'), fname_2)
        ]
        
        # Key must be 'waveform_files' as updated in index.html/app.py
        data = {'waveform_files': files_to_upload}
        
        print("Uploading multiple files via /upload...")
        rv = self.client.post('/upload', data=data, follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        
        # 2. Check Session
        with self.client.session_transaction() as sess:
            # Check deconv_files_info (reused key)
            files_info = sess.get('deconv_files_info', [])
            print("Files in session:", [f['filename'] for f in files_info])
            self.assertEqual(len(files_info), 2)
            self.assertIn(fname_1, [f['filename'] for f in files_info])
            self.assertIn(fname_2, [f['filename'] for f in files_info])
            
            # Check history (Waveform Viewer should detect 1st file)
            history = sess.get('history', [])
            self.assertTrue(len(history) > 0)
            print("History Step 1:", history[0]['action'])
            
        # 3. Test Deconv Process (using these files)
        # We need to set role/depth for them.
        process_data = {
            f'role_{fname_1}': 'reference',
            f'depth_{fname_1}': '0',
            f'type_{fname_1}': 'borehole',
            f'role_{fname_2}': 'interest',
            f'depth_{fname_2}': '50',
            f'type_{fname_2}': 'building',
            'amp_factor': '1.0',
            't_min': '-5.0',
            't_max': '5.0',
            'dstack': '10',
            'r': '1',
            'sp': '1'
        }
        
        print("Running /deconv_process...")
        rv = self.client.post('/deconv_process', data=process_data, follow_redirects=True)
        self.assertEqual(rv.status_code, 200)
        
        # Check for Canvas or Result
        self.assertIn(b'Interactive Selector', rv.data)
        self.assertIn(b'deconvCanvas', rv.data)

if __name__ == '__main__':
    unittest.main()
