import tensorflow as tf

model_path = "../Models/2021-12-03_09-37-05.h5"

model = tf.keras.models.load_model(model_path)
model.summary()
# updated_model = ResNetLike()
# updated_model.create_model((8, 8))
# updated_model.set_weights(model.get_weights())
# updated_model.save_model(model_path)
