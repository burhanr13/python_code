import tensorflow as tf

model = tf.keras.models.load_model("sentiment_model_mse",compile=False)

def test(text):
    tf.print("sentiment value: ",model(tf.convert_to_tensor([text]))[0,0])

while True:
    text = input("enter text, sentiment value 0(negative) to 1(positive), q to quit: ")
    if text == 'q':
        break
    test(text)