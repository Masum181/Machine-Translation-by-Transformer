import streamlit as st

st.header("Machine Translation by Transformer.")
st.write(
    """**Transformer** basically contain **Encoder** and **Decoder** section. Also ***Attention Layer**.
    This page contains all about transformer. And by this Transformer we train a translation model. Then we will try this code."""
)

st.image("https://3.bp.blogspot.com/-aZ3zvPiCoXM/WaiKQO7KRnI/AAAAAAAAB_8/7a1CYjp40nUg4lKpW7covGZJQAySxlg8QCLcBGAs/s1600/transform20fps.gif", caption="**Transformer**")

st.image("https://www.tensorflow.org/images/tutorials/transformer/transformer.png", caption="**Transformer Architecture**")
st.code(
    """
import logging
import time

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow as tf

import tensorflow_text
"""
)

st.write("### Load and download the dataset")

st.code(
    """examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True, 
                               as_supervised=True)

train_examples, val_examples = examples['train'], examples['validation']
"""
)

st.write(
    """### SetUp tokenizer
    Now that you have loaded the dataset, you need to tokenize the text, so that each element is represented as a token or token ID (a numeric representation)."""
)

st.code(
    """
model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)
"""
)

st.write("### Setup data Pipeline")

st.code(
    """
MAX_TOKENS=128
def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    return (pt, en_inputs), en_labels
    """
)
st.code(
    """
BUFFER_SIZE = 20000
BATCH_SIZE = 64

def make_batches(ds):
  return (
      ds
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .map(prepare_batch, tf.data.AUTOTUNE)
      .prefetch(buffer_size=tf.data.AUTOTUNE))
      
"""
)

st.write("### Test the Dataset")
st.code(
    """
train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

for (pt, en), en_labels in train_batches.take(1):
    break
print(pt.shape)
print(en.shape)
print(en_labels.shape)

"""
)

st.write("### Embedding and Positional Embedding")
st.code(
    """
def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis] # (seq, 1)
    depths = np.arange(depth) [np.newaxis, :] / depth # (1, depth)
    
    angle_rates = 1/(10000 ** depths) # (1, depth)
    angle_rads = positions * angle_rates # (pos, depth)

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )
    return tf.cast(pos_encoding, dtype=tf.float32)

"""
)

st.code(
    """
class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model =d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=d_model)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, : length, :]
        return x
"""
)

st.write(
    """### Attention Layer"""
)

st.code(
    """
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        
"""
)

st.code(
    """
class CrossAttention(BaseAttention):
    def call(self, x, context):
        attn_output, attn_scores = self.mha(query=x,
                                            key=context,
                                            value=context,
                                            return_attention_scores=True)
        # cache the attention scores for ploting later
        self.last_attn_scores = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
"""
)

st.code(
    """
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x= self.add([x, attn_output])
        x = self.layernorm(x)
        return x
"""
)

st.code(
    """
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            use_causal_mask = True)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
"""
)

st.write(
    """### The Feed Forward Network"""
)

st.code(
    """
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, diff, dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(diff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()
    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x)
        return x"""
)


st.write(
    """### Encoder"""
)
st.code(
    """
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, * , d_model, num_heads, diff, dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout = dropout_rate
        )
        self.ffn = FeedForward(d_model, diff)

    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
"""
)

st.code(
    """
class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, diff, vocab_size, dropout_rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(
            vocab_size = vocab_size, d_model = d_model
        )
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         diff=diff,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        # shape of x (batch, seq_len)
        x = self.pos_embedding(x) # shape (batch, seq_len, d_model)

        # add dropout
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        return x # shape (batch, seq_len, d_model)
"""
)

st.write(
    """### Decoder"""
)

st.code(
    """
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, diff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()

        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.cross_attention = CrossAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate
        )

        self.ffn = FeedForward(d_model, diff)

    def call(self, x, context):
        x = self.causal_self_attention(x=x)
        x = self.cross_attention(x=x, context=context)

        # Cache the last attention scores for plotting later
        self.last_attn_scores = self.cross_attention.last_attn_scores
        x = self.ffn(x)
        return x
"""
)

st.code(
    """
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, diff, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                                d_model=d_model)
        self.dropout= tf.keras.layers.Dropout(dropout_rate)
        self.dec_layers = [
            DecoderLayer(d_model=d_model, num_heads=num_heads,diff=diff, dropout_rate=dropout_rate)
            for _ in range(num_layers)
        ]
        self.last_attn_scores = None
    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x,context)

        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        return x"""
)

st.write(
    """### Transformer"""
)
st.code(
    """
class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, diff, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                              num_heads=num_heads, diff=diff,
                              vocab_size=input_vocab_size,
                              dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                              num_heads=num_heads, diff=diff,
                              vocab_size=target_vocab_size,
                              dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context) # (batch, context_len, d_model)
        x = self.decoder(x, context) # batch,target_lan, d_model

        # final linear layer output
        logits = self.final_layer(x)

        try:
            del logits._keras_mask
        except AttributeError:
            pass
        return logits"""
)

st.write("### Testing the transformer")
st.code(
    """
num_layers = 4
d_model = 128
diff = 512
num_heads = 8
dropout_rate=0.1

transformer = Transformer(num_layers=num_layers,
                         d_model=d_model,
                         num_heads=num_heads,
                         diff=diff,
                         input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
                         target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
                         dropout_rate=dropout_rate)
"""
)
st.code(
    """
output = transformer((pt,en))
print(en.shape)
print(pt.shape)
print(output.shape)"""
)

st.write("### Training.")
st.code(
    """
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps
    def __call__(self, step):
        step =tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
"""
)

st.code(
    """
learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
"""
)

st.code(
    """
def masked_loss(label, pred):
  mask = label != 0
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
  loss = loss_object(label, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss *= mask

  loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
  return loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)
  
  """
)

st.code(

    """
transformer.compile(loss=masked_loss, optimizer=optimizer, metrics=[masked_accuracy])
transformer.fit(train_batches, epochs=3, validation_data=val_batches)
"""
)

st.write("### Translator")
st.code(
    """
class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
    self.tokenizers = tokenizers
    self.transformer = transformer

  def __call__(self, sentence, max_length=MAX_TOKENS):
    assert isinstance(sentence, tf.Tensor)
    if len(sentence.shape) == 0:
      sentence = sentence[tf.newaxis]

    sentence = self.tokenizers.pt.tokenize(sentence).to_tensor()

    encoder_input = sentence
    start_end = self.tokenizers.en.tokenize([''])[0]
    start = start_end[0][tf.newaxis]
    end = start_end[1][tf.newaxis]

    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)

    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
      predictions = self.transformer([encoder_input, output], training=False)

      predictions = predictions[:, -1:, :] 

      predicted_id = tf.argmax(predictions, axis=-1)
      output_array = output_array.write(i+1, predicted_id[0])

      if predicted_id == end:
        break

    output = tf.transpose(output_array.stack())
    text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.

    tokens = tokenizers.en.lookup(output)[0]

    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return text, tokens, attention_weights"""
)
st.code("translator = Translator(tokenizers, transformer)")
st.code(
    """
def print_translation(sentence, tokens, ground_truth):
  print(f'{"Input:":15s}: {sentence}')
  print(f'{"Prediction":15s}: {tokens.numpy().decode("utf-8")}')
  print(f'{"Ground truth":15s}: {ground_truth}')"""
)

st.code(
    """sentence = 'este é um problema que temos que resolver.'
ground_truth = 'this is a problem we have to solve .'

translated_text, translated_tokens, attention_weights = translator(
    tf.constant(sentence))

print_translation(sentence, translated_text, ground_truth)"""
)

