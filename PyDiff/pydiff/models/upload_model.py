import os
import dropbox
from dropbox.exceptions import ApiError
from threading import Thread
from basicsr.utils import get_root_logger

class ModelUploader():

    def __init__(self, opt):
        self.chunk_size=4*1024*1024
        self.access_token = os.environ.get('DROPBOX_TOKEN')
        self.experiment_name = opt['name']
        self.opt = opt
        self.logger = get_root_logger()

    def chunked_upload_to_dropbox(self, filepath, dropbox_path):
        access_token = self.access_token
        filepath = os.path.abspath(filepath)
        dbx = dropbox.Dropbox(access_token)
        
        file_size = os.path.getsize(filepath)
        with open(filepath, 'rb') as f:
            if file_size <= self.chunk_size:
                try:
                    dbx.files_upload(f.read(), dropbox_path, mode=dropbox.files.WriteMode.overwrite)
                    print(f"Uploaded {filepath} successfully.")
                except ApiError as e:
                    print(f"API error during upload: {e}")
            else:
                upload_session_start_result = dbx.files_upload_session_start(f.read(self.chunk_size))
                cursor = dropbox.files.UploadSessionCursor(session_id=upload_session_start_result.session_id, offset=f.tell())
                commit = dropbox.files.CommitInfo(path=dropbox_path, mode=dropbox.files.WriteMode.overwrite)

                while f.tell() < file_size:
                    if (file_size - f.tell()) <= self.chunk_size:
                        dbx.files_upload_session_finish(f.read(self.chunk_size), cursor, commit)
                    else:
                        dbx.files_upload_session_append_v2(f.read(self.chunk_size), cursor)
                        cursor.offset = f.tell()
                print(f"Chunked upload for {filepath} completed successfully.")

    def _upload_latest_model(self):
        self.logger.info(f'Thread started: upload_latest_model')
        model_path = os.path.join("../experiments/", self.experiment_name, "models/net_g_latest.pth")
        model_path = os.path.abspath(model_path)

        if os.path.exists(model_path):
            self.chunked_upload_to_dropbox(model_path, '/ClearOcean/latest.pth')
        self.logger.info(f'Thread started: upload_latest_model done')

    def _upload_best_ssim_model(self):
        self.logger.info(f'Thread started: upload_best_ssim_model')
        model_path = os.path.join("../experiments/", self.experiment_name, "models/net_g_best_ssim.pth")
        model_path = os.path.abspath(model_path)
        
        if os.path.exists(model_path):
            self.chunked_upload_to_dropbox(model_path, '/ClearOcean/best_ssim.pth')
        self.logger.info(f'Thread started: upload_best_ssim_model done')

    def _upload_best_psnr_model(self):
        self.logger.info(f'Thread started: upload_best_psnr_model')
        model_path = os.path.join("../experiments/", self.experiment_name, "models/net_g_best_psnr.pth")
        model_path = os.path.abspath(model_path)
        
        if os.path.exists(model_path):
            self.chunked_upload_to_dropbox(model_path, '/ClearOcean/best_psnr.pth')
        self.logger.info(f'Thread started: upload_best_psnr_model done')

            
    def upload_latest_model(self):
        self.logger.info(f'Thread start: upload_latest_model')
        thread = Thread(target = self._upload_latest_model)
        thread.start()

    def upload_best_ssim_model(self):
        self.logger.info(f'Thread start: upload_best_ssim_model')
        thread = Thread(target = self._upload_best_ssim_model)
        thread.start()
        

    def upload_best_psnr_model(self):
        self.logger.info(f'Thread start: upload_best_psnr_model')
        thread = Thread(target = self._upload_best_psnr_model)
        thread.start()
        

    # def upload_run_logs(self):
    #     run_log_path = os.path.join("../experiments/", self.experiment_name, "models/net_g_latest.pth")
    #     self.chunked_upload_to_dropbox(run_log_path, '/ClearOcean/run.log')


