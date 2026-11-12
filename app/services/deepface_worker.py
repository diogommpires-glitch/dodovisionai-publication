import multiprocessing as mp
import time
import numpy as np

class DeepFaceWorker:
    """
    Independent process worker for DeepFace inference.
    Moves heavy TensorFlow/PyTorch inference out of the main thread pool.
    """
    def __init__(self, max_queue_size=10):
        self.input_queue = mp.Queue(maxsize=max_queue_size)
        self.output_queue = mp.Queue(maxsize=max_queue_size)
        self.process = mp.Process(target=self._run, daemon=True)
        self.started = False
        
    def start(self):
        if not self.started:
            self.process.start()
            self.started = True

    def _run(self):
        import traceback
        import cv2
        # Import inside the worker process to isolate the ML context
        import detectors
        
        # Optionally wait for warming up
        print("[DeepFaceWorker] Process started")
        
        while True:
            try:
                task = self.input_queue.get()
                if task is None:
                    break  # Stop signal
                    
                job_id, face_idx, crop_bytes, shape, bs_emotion = task
                try:
                    # Decode crop
                    if crop_bytes is not None:
                        crop = np.frombuffer(crop_bytes, dtype=np.uint8).reshape(shape)
                    else:
                        crop = None
                        
                    df = detectors.run_deepface(crop)
                    self.output_queue.put({
                        "job_id": job_id,
                        "face_idx": face_idx, 
                        "result": df,
                        "bs_emotion": bs_emotion,
                        "error": None
                    })
                except Exception as e:
                    self.output_queue.put({
                        "job_id": job_id,
                        "face_idx": face_idx, 
                        "result": None,
                        "bs_emotion": bs_emotion,
                        "error": str(e)
                    })
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"[DeepFaceWorker] Loop error: {e}")
                
    def submit(self, job_id, face_idx, crop, bs_emotion=None):
        try:
            # We serialize the numpy array to avoid pickling overhead issues
            if crop is not None:
                crop_bytes = crop.tobytes()
                shape = crop.shape
            else:
                crop_bytes = None
                shape = None
            
            self.input_queue.put_nowait((job_id, face_idx, crop_bytes, shape, bs_emotion))
            return True
        except mp.queues.Full:
            return False

    def get_results(self):
        results = []
        while not self.output_queue.empty():
            try:
                results.append(self.output_queue.get_nowait())
            except Exception:
                break
        return results

    def stop(self):
        try:
            if self.started:
                self.input_queue.put(None)
                self.process.join(timeout=2)
                if self.process.is_alive():
                    self.process.terminate()
                self.started = False
        except Exception:
            pass

worker_instance = DeepFaceWorker()
