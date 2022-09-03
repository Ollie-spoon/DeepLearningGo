"""
This was originally not created by Oliver Nicholls, simply taken straight from the
dlgo github. However, none of the code actually worked so it has been rewritten in
a more palatable format.


"""
import tempfile
import os

import h5py
from keras import backend
from tensorflow.keras.models import load_model, save_model


# def save_model_to_hdf5_group(model, f):
#     # Use Keras save_model to save the full model (including optimizer
#     # state) to a file.
#     # Then we can embed the contents of that HDF5 file inside ours.
#     tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
#     try:
#         # os.close(tempfd)
#         model.save(tempfname)
#         os.close(tempfd)
#         serialized_model = h5py.File(tempfname, 'r')
#         root_item = serialized_model.get('/')
#         serialized_model.copy(root_item, f, 'kerasmodel')
#         serialized_model.close()
#     finally:
#         os.unlink(tempfname)
#
#
# def load_model_from_hdf5_group(f, custom_objects=None):
#     # Extract the model into a temporary file. Then we can use Keras
#     # load_model to read it.
#     tempfd, tempfname = tempfile.mkstemp(prefix='tmp-kerasmodel')
#     try:
#         os.close(tempfd)
#         serialized_model = h5py.File(tempfname, 'w')
#         root_item = f.get('kerasmodel')
#         for attr_name, attr_value in root_item.attrs.items():
#             serialized_model.attrs[attr_name] = attr_value
#         for k in root_item.keys():
#             f.copy(root_item.get(k), serialized_model, k)
#         serialized_model.close()
#         return load_model(tempfname, custom_objects=custom_objects)
#     finally:
#         os.unlink(tempfname)

def save_model(model, file_directory: str = None, network_name="", encoder_name="", training_steps: str = None):
    """
    Used to save tensorflow models to a hdf5 file following the format:
      'network_name'_'encoder_name'_'training_steps'.h5

    :param model: Any keras model.
    :param file_directory: The relative path of the directory for the target file.
    Leave blank if target directory is the current working directory.
    :param network_name: Name of the model used.
    :param encoder_name: Name of the encoder used.
    :param training_steps: Number of training steps used so far.
    :return:
    """
    file_directory = "" if file_directory is None else "\\" + file_directory
    file_directory += "\\" + network_name.lower() + "_" + encoder_name.lower()
    if training_steps is not None:
        file_directory += "_" + str(training_steps)
    file_directory += ".h5"
    cwd = os.getcwd()
    path = cwd + file_directory
    print("path: " + str(path))
    model.save(path, save_format='h5')


def load_keras_model(file_name: str):
    """
    Load a keras model from a .h5 file.

    :param file_name: The name of a file, must include the '.h5' file extension at the end.
    :return:
    """
    assert file_name[-3:] == ".h5"
    assert file_name != ".h5"

    "format name"
    split = file_name[:-3].split("_")
    network_name = split[0].lower()
    encoder_name = split[1].lower()

    with h5py.File(file_name, 'r') as h5file:
        model = load_model(h5file)

    output_dim = model.layers[-1].output_shape[1]

    return model, network_name, encoder_name, output_dim


def set_gpu_memory_target(frac):
    """Configure Tensorflow to use a fraction of available GPU memory.
    Use this for evaluating models in parallel. By default, Tensorflow
    will try to map all available GPU memory in advance. You can
    configure to use just a fraction so that multiple processes can run
    in parallel. For example, if you want to use 2 works, set the
    memory fraction to 0.5.
    If you are using Python multiprocessing, you must call this function
    from the *worker* process (not from the parent).
    This function does nothing if Keras is using a backend other than
    Tensorflow.
    """
    if backend.backend() != 'tensorflow':
        return
    # Do the import here, not at the top, in case Tensorflow is not
    # installed at all.
    import tensorflow as tf
    from keras.backend import set_session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = frac
    set_session(tf.compat.v1.Session(config=config))
