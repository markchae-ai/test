import os, tempfile, json
from google.cloud import storage
import module

storage_client = storage.Client()

def exec_module_fuction(temp_file, temp_type):
    module.file_open(temp_file, temp_type)
    module.get_array_noise()
    module.stereo_judgement()
    module.phase_judgement()
    module.dic()

def download_from_storage(bucket, name, ext):
    _, temp_local_filename = tempfile.mkstemp(suffix=ext)        
    blob = storage_client.bucket(bucket).get_blob(name)
    blob.download_to_filename(temp_local_filename)
    return temp_local_filename

def audio_judgement(data):
    file_name = data['path']
    _, file_extension = os.path.splitext(data['path'])
    tmp_file = download_from_storage(data['bucket_name'], file_name, file_extension)
    exec_module_fuction(tmp_file, data['type'])
    os.remove(tmp_file)

def start_function(request):
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    audio_judgement(request.get_json()["data"])
    return (json.dumps(module.dic()), 200, headers)