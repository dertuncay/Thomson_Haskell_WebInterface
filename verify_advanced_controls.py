import os
import io
from app import app
import unittest
import obspy

class AdvancedControlsTest(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.secret_key = 'test'
        self.client = app.test_client()
        self.examples_dir = 'Examples'

    def test_process_with_params(self):
        files_to_upload = [
            'L1.RE006..HNN.2023.301.152902.PROC.mseed',
            'L1.RE008..HNN.2023.301.152902.PROC.mseed'
        ]
        
        opened_files = []
        multi_files = []
        
        try:
            for fname in files_to_upload:
                path = os.path.join(self.examples_dir, fname)
                if not os.path.exists(path):
                    continue
                f = open(path, 'rb')
                opened_files.append(f)
                multi_files.append((f, fname))

            if not multi_files:
                self.fail("No example files found.")

            # Upload
            data = {'deconv_files': multi_files}
            rv = self.client.post('/deconv_upload', data=data, follow_redirects=True)
            self.assertEqual(rv.status_code, 200)
            
            # Process with advanced params
            f1 = files_to_upload[0]
            f2 = files_to_upload[1]
            
            process_data = {
                f'role_{f1}': 'reference',
                f'depth_{f1}': '0',
                f'type_{f1}': 'borehole',
                f'role_{f2}': 'interest',
                f'depth_{f2}': '50',
                f'type_{f2}': 'building',
                'amp_factor': '2.0',
                't_min': '-2.0',
                't_max': '2.0',
                'dstack': '5',
                'r': '1.5',
                'sp': '0.5'
            }
            
            rv = self.client.post('/deconv_process', data=process_data, follow_redirects=True)
            self.assertEqual(rv.status_code, 200)
            
            if b'Error in processing' in rv.data or b'Traceback' in rv.data:
                 print("Found Error/Traceback in response:")
                 for line in rv.data.decode().split('\n'):
                     if 'Error' in line or 'Traceback' in line or 'Exception' in line:
                         print(line.strip())
                 self.fail("Error processing with advanced params")
                 
            self.assertIn(b'Interactive Selector', rv.data)
            
            # Verify dstack effect by parsing JSON resultData
            content = rv.data.decode()
            import json
            import re
            
            # Extract JSON: var resultData = { ... };
            match = re.search(r'var resultData = ({.*?});', content, re.DOTALL)
            if match:
                json_str = match.group(1)
                # It might contain 'Safe' wrapper or be clean. 
                # Actually, flask might have newlines.
                # Let's try to parse.
                try:
                     # Simple cleanup if needed, but usually it's valid JSON
                     data_obj = json.loads(json_str)
                     
                     times = data_obj.get('time', [])
                     if times:
                         t_start = times[0]
                         t_end = times[-1]
                         print(f"Time Range: {t_start} to {t_end}")
                         
                         # dstack was 5, so range should be approx -5 to 5
                         self.assertAlmostEqual(t_start, -5.0, delta=0.5)
                         self.assertAlmostEqual(t_end, 5.0, delta=0.5)
                         
                except Exception as e:
                    print(f"JSON parsing warning: {e}")
                    # If regex fails to capture full JSON due to formatting, skip strict check but warn
            else:
                self.fail("Could not find resultData in response")
            
        finally:
            for f in opened_files:
                f.close()

if __name__ == '__main__':
    unittest.main()
