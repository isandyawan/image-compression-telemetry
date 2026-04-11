import tensorflow as tf
import tensorflow_compression as tfc
from glob import glob

##process the image into suitable dimension
class AutoEncoder():
    model = None
    dtypes = None
    
    def __init__(self, model_path ):
        self.model = self.load_model(model_path)
        self.dtypes = [t.dtype for t in self.model.decompress.input_signature]
        
    def load_img(self, path):
        string = tf.io.read_file(path)
        image = tf.image.decode_image(string, channels=3)
        return image

    def load_model(self, model_path):
        model = tf.keras.models.load_model(model_path,compile=False)
        return model
    
    def compress_tensor(self, bytes_data):
        image = tf.image.decode_image(bytes_data, channels=3)
        compressed = self.model.compress(image)
        packed = tfc.PackedTensors()
        packed.pack(compressed)
        return packed.string

    def compress(self, path):
        image = self.load_img(path)
        compressed = self.model.compress(image)
        packed = tfc.PackedTensors()
        packed.pack(compressed)

        return packed.string

    def decompress(self, packed_string):
        
        packed = tfc.PackedTensors(packed_string)        
        tensors = packed.unpack(self.dtypes)
        x_hat = self.model.decompress(*tensors)
        png_bytes = tf.image.encode_png(x_hat)
        return png_bytes.numpy()
    
    def bytes_to_tensor(self, img_bytes):
        # Decode bytes (PNG/JPG) jadi tensor
        image = tf.image.decode_image(img_bytes, channels=3)
        # Konversi ke float32 dan normalisasi ke range [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image
    
    def calculate_metrics_from_bytes(self, original_bytes, reconstructed_bytes):
        # 1. Decode keduanya jadi tensor piksel
        orig_tensor = self.bytes_to_tensor(original_bytes)
        recon_tensor = self.bytes_to_tensor(reconstructed_bytes)
        
        # Pastikan ukurannya sama (karena MSE butuh tensor yang identik dimensinya)
        # Jika perlu, resize salah satu ke ukuran yang lain
        recon_tensor = tf.image.resize(recon_tensor, [orig_tensor.shape[0], orig_tensor.shape[1]])

        # 2. Hitung MSE
        mse = tf.reduce_mean(tf.square(orig_tensor - recon_tensor))
        
        # 3. Hitung Fidelity (%)
        fidelity = (1.0 - mse) * 100
        
        # 4. Hitung PSNR (dB)
        psnr = tf.image.psnr(orig_tensor, recon_tensor, max_val=1.0)

        return fidelity.numpy()

