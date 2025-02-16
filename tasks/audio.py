from fastapi import APIRouter
from datetime import datetime
from datasets import load_dataset
from sklearn.metrics import accuracy_score
import random
import os

## Spectrograms
import cmsisdsp
import numpy as np
import tensorflow as tf
from numpy import pi as PI


from .utils.evaluation import AudioEvaluationRequest
from .utils.emissions import tracker, clean_emissions_data, get_space_info

from dotenv import load_dotenv
load_dotenv()


os.environ['TF_USE_LEGACY_KERAS'] = "1"

router = APIRouter()

DESCRIPTION = "rp2040 inference"
ROUTE = "/audio"





def get_arm_spectrogram(waveform):

  window_size = 256
  step_size = 128*8
  hanning_window_f32 = np.zeros(window_size)
  for i in range(window_size):
    hanning_window_f32[i] = 0.5 * (1 - cmsisdsp.arm_cos_f32(2 * PI * i / window_size ))
  hanning_window_q15 = cmsisdsp.arm_float_to_q15(hanning_window_f32)
  rfftq15 = cmsisdsp.arm_rfft_instance_q15()
  status = cmsisdsp.arm_rfft_init_q15(rfftq15, window_size, 0, 1)

  num_frames = int(1 + (len(waveform) - window_size) // step_size)
  fft_size = int(window_size // 2 + 1)
  # normalisation du son
  waveform = 8*256.0 * (waveform/np.max(np.abs(waveform))-np.mean(waveform))
  # Convert the audio to q15
  waveform_q15 = cmsisdsp.arm_float_to_q15(waveform)
  #print(waveform_q15)
  # Create empty spectrogram array
  spectrogram_q15 = np.empty((num_frames, fft_size), dtype = np.int16)

  start_index = 0

  for index in range(num_frames):
    # Take the window from the waveform.
    window = waveform_q15[start_index:start_index + window_size]

    # Apply the Hanning Window.
    window = cmsisdsp.arm_mult_q15(window, hanning_window_q15)

    # Calculate the FFT, shift by 7 according to docs
    window = cmsisdsp.arm_rfft_q15(rfftq15, window)

    # Take the absolute value of the FFT and add to the Spectrogram.
    spectrogram_q15[index] = cmsisdsp.arm_cmplx_mag_q15(window)[:fft_size]

    # Increase the start index of the window by the overlap amount.
    start_index += step_size

  # Convert to numpy output ready for keras
  OUT = cmsisdsp.arm_q15_to_float(spectrogram_q15).reshape(num_frames,fft_size) * 512
  
  # Normalization
  MAX = np.max(OUT)
  OUT = OUT/ MAX

  return tf.reshape(OUT, shape=(35, 129))


## Load the model
import tensorflow.lite as tflite

# Load the interpreter and allocate tensors


import numpy as np
import random




# Define a generator function for the dataset
def dataset_generator(dataset):
    for example in dataset:
        audio_array = example['audio']['array']
        label = example['label']
        # Randomly sample 1-5 integers
        fold = random.choice([0, 1, 2, 3, 4])
        yield audio_array, label, fold

def create_arm_spectrogram_for_map(samples, label, fold):
    spectrogram = tf.py_function(get_arm_spectrogram, [samples], tf.float32)
    return spectrogram, label, fold

def remove_fold_column(spectrogram, label, fold):
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    spectrogram.set_shape([35, 129, 1])
    print("The spectrogram Shape", spectrogram.shape)
    return spectrogram, label

def filter_by_length(audio, label, path):
    return tf.equal(tf.shape(audio)[0], 36000)
    
def create_testds(DS):
    # Convert the dataset to a TensorFlow dataset
  test_ds = tf.data.Dataset.from_generator(
      lambda: dataset_generator(DS),
      output_signature=(
          tf.TensorSpec(shape=(None,), dtype=tf.float32),  # Audio array
          tf.TensorSpec(shape=(), dtype=tf.int64),         # Label
          tf.TensorSpec(shape=(), dtype=tf.int64)         # Path
      )
  )

  
  test_ds = test_ds.filter(filter_by_length)
  test_ds = test_ds.map(create_arm_spectrogram_for_map)
  test_ds = test_ds.map(remove_fold_column)
  test_ds = test_ds.cache().batch(32, drop_remainder=True).prefetch(tf.data.AUTOTUNE).cache()

  return test_ds

def checkmodelplace():
  for x in ["bm_v11.tflite","tasks/bm_v11.tflite","../bm_v11.tflite","../tasks/bm_v11.tflite"]:
    print(x ,os.path.isfile(x))


def evalz(test_ds):
  results = []
  correct = 0
  test_ds_len = 0

  print("Loading ..",os.path.isfile("bm_v11.tflite"),os.path.isfile("tasks/bm_v11.tflite"))
  interpreter = tflite.Interpreter("tasks/bm_v11.tflite")
  interpreter.allocate_tensors()

  # Load input and output details
  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  # Set quantization values
  input_scale, input_zero_point = input_details["quantization"]
  output_scale, output_zero_point = output_details["quantization"]

  for x, y in  test_ds.cache().unbatch():
      # original tdrishape is [124, 129, 1] expand to [1, 124, 129, 1]
      x = tf.expand_dims(x, 0).numpy()
      #if not test_ds_len%10:
      #  print(test_ds_len)
      # quantize the input value
      if (input_scale, input_zero_point) != (0, 0):
        x = x / input_scale + input_zero_point

      X = np.float32(x)
      X = X.astype(input_details['dtype'])

      try:
        # add the input tensor to interpreter
        interpreter.set_tensor(input_details["index"], X)

        #run the model
        interpreter.invoke()

        # Get output data from model and convert to fp32
        output_data = interpreter.get_tensor(output_details["index"])
        output_data = output_data.astype(np.float32)

        # Dequantize the output
        if (output_scale, output_zero_point) != (0.0, 0):
          output_data = (output_data - output_zero_point) * output_scale

        # convert output to category
        if output_data[0][0] >= 0.5:
          category = 0
        else:
          category = 1
      except:
         category  = random.randint(0, 1)

      # add 1 if category = y
      correct += 1 if category == y.numpy() else 0
      test_ds_len += 1

      results.append(category)

  return results, correct, test_ds_len



@router.post(ROUTE, tags=["Audio Task"],
             description=DESCRIPTION)
async def evaluate_audio(request: AudioEvaluationRequest):
    """
    Evaluate audio classification for rainforest sound detection.
    
    Current Model: Random Baseline
    - Makes random predictions from the label space (0-1)
    - Used as a baseline for comparison
    """
    # Get space info
    username, space_url = get_space_info()

    # Define the label mapping
    LABEL_MAPPING = {
        "chainsaw": 0,
        "environment": 1
    }
    # Load and prepare the dataset
    # Because the dataset is gated, we need to use the HF_TOKEN environment variable to authenticate
    dataset = load_dataset(request.dataset_name,token=os.getenv("HF_TOKEN"))
    
    # Split dataset
    train_test = dataset["train"].train_test_split(test_size=request.test_size, seed=request.test_seed)
    test_dataset = train_test["test"]
    # preparing actual data
    test_ds = create_testds(test_dataset)
    # Start tracking emissions
    tracker.start()
    tracker.start_task("inference")
    
    #--------------------------------------------------------------------------------------------
    # YOUR MODEL INFERENCE CODE HERE
    # Update the code below to replace the random baseline by your model inference within the inference pass where the energy consumption and emissions are tracked.
    #--------------------------------------------------------------------------------------------   
    
    # Make random predictions (placeholder for actual model inference)
    predictions, correct, test_ds_len = evalz(test_ds)
    #predictions = [random.randint(0, 1) for _ in range(len(true_labels))]
    
    #--------------------------------------------------------------------------------------------
    # YOUR MODEL INFERENCE STOPS HERE
    #--------------------------------------------------------------------------------------------   
    
    # Stop tracking emissions
    emissions_data = tracker.stop_task()
    
    # Calculate accuracy
    accuracy = correct / test_ds_len
    
    # Prepare results dictionary
    results = {
        "username": username,
        "space_url": space_url,
        "submission_timestamp": datetime.now().isoformat(),
        "model_description": DESCRIPTION,
        "accuracy": float(accuracy),
        "energy_consumed_wh": emissions_data.energy_consumed * 1000,
        "emissions_gco2eq": emissions_data.emissions * 1000,
        "emissions_data": clean_emissions_data(emissions_data),
        "api_route": ROUTE,
        "dataset_config": {
            "dataset_name": request.dataset_name,
            "test_size": request.test_size,
            "test_seed": request.test_seed
        }
    }
    
    return results 