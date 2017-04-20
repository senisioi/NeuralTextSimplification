import requests
import logging

logging.basicConfig(format = u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level = logging.NOTSET)

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    logging.info("Downloading "+id + " to "+destination)
    logging.info("Please be patient, it may take a while...")
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)
    logging.info("...")
    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    
    logging.info("Done with " + id)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

'''
    to download the .t7 NTS models used for text simplification
    if for some reason, this doanload fails, please use the direct urls:
    - for NTS:
        https://drive.google.com/file/d/0B_pjS_ZjPfT9QjFsZThCU0xUTnM

    -for NTS-w2v:
        https://drive.google.com/file/d/0B_pjS_ZjPfT9U1pJNy1UdV9nNk0

'''

if __name__ == "__main__":
    NTS_model = '0B_pjS_ZjPfT9QjFsZThCU0xUTnM' 
    NTS_model_output = 'NTS_epoch11_10.19.t7'
    download_file_from_google_drive(NTS_model, NTS_model_output)

    NTS_w2v_model = '0B_pjS_ZjPfT9U1pJNy1UdV9nNk0' 
    NTS_w2v_model_output = 'NTS-w2v_epoch11_10.20.t7'
    download_file_from_google_drive(NTS_w2v_model, NTS_w2v_model_output)