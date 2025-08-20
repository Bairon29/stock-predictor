from __future__ import print_function
import os.path
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaIoBaseDownload
import pickle
import io

# If modifying scopes, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def authenticate_drive():
    """Authenticate and return Google Drive service object."""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'google_config.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('drive', 'v3', credentials=creds)
    return service

def upload_file(service, file_path, drive_folder_id=None):
    file_metadata = {'name': os.path.basename(file_path)}
    if drive_folder_id:
        file_metadata['parents'] = [drive_folder_id]

    media = MediaFileUpload(file_path, resumable=True)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    print(f"Uploaded {file_path} with File ID: {file.get('id')}")
    return file.get('id')

def download_file(service, file_path, file_id):
    request = service.files().get_media(fileId=file_id)
    with open(file_path, 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            print(f"Download {int(status.progress() * 100)}%.")

def download_file_to_variable(service, file_id):
    file_buffer = io.BytesIO()
    request = service.files().get_media(fileId=file_id)
    downloader = MediaIoBaseDownload(file_buffer, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
    file_buffer.seek(0)
    return file_buffer.read()
            
# === Example Usage ===
if __name__ == "__main__":
    pass
    # service = authenticate_drive()
    # Upload a file
    # upload_file(service, 'google_drive/example.txt', drive_folder_id='13psIjHiLQsottsNjP3WGLaXiFMPTt-38')
    # Download a file
    # download_file(service, 'mmmm', '13psIjHiLQsottsNjP3WGLaXiFMPTt-38')