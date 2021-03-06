import tensorflow as tf

from AmoebaPlayGround.Agents.TensorflowModels import ResNetLike

model_path = "Models/2021-12-03_09-37-05.h5"

model = tf.keras.models.load_model(model_path)

updated_model = ResNetLike(6).create_model((8, 8))
updated_model.set_weights(model.get_weights())
updated_model.save(model_path)
