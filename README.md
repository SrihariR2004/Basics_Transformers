# Basics_Transformers

The purpose of the above code, demonstrated with a simplified example, is to show the working of the self-attention mechanism in a Transformer model. Let's break it down step-by-step with an example in math to better understand how values are assigned and processed.

#Key Concepts:

Self-Attention: The self-attention mechanism allows a model to focus on different words in a sentence when making predictions, even if they are far apart. In simpler terms, self-attention measures how much each word in a sentence should "pay attention" to every other word when creating a representation of that word.

Q (Query), K (Key), V (Value): These are three vectors that represent each word in the sentence. The relationship between these vectors is computed to determine the attention score.

Next Word Prediction: After processing the sentence with self-attention, the model predicts the next word by analyzing the context.

Mathematical Explanation Using an Example
Example Sentence:
"I love coding."

#Step 1: Embedding Words into Vectors

Each word in the sentence is converted into a numerical vector (embedding). For simplicity, assume each word is represented by a 2-dimensional vector.

I → [0.1, 0.2]

love → [0.3, 0.4]

coding → [0.5, 0.6]

These are the word embeddings.

#Step 2: Generating Q, K, V Vectors

The next step is to generate three new vectors, Query (Q), Key (K), and Value (V), for each word in the sentence by multiplying the word embedding with three randomly initialized weight matrices (W_q, W_k, W_v). Let's assume the matrices for this simple case are:

W_q = [[0.1, 0.2], [0.3, 0.4]]

W_k = [[0.5, 0.6], [0.7, 0.8]]

W_v = [[0.9, 1.0], [1.1, 1.2]]

For each word, we calculate:

For 'I':

Q_I = [0.1, 0.2] @ W_q = [0.10.1 + 0.20.3, 0.10.2 + 0.20.4] = [0.07, 0.1]

K_I = [0.1, 0.2] @ W_k = [0.10.5 + 0.20.7, 0.10.6 + 0.20.8] = [0.19, 0.22]

V_I = [0.1, 0.2] @ W_v = [0.10.9 + 0.21.1, 0.11.0 + 0.21.2] = [0.37, 0.44]

For 'love':

Q_love = [0.3, 0.4] @ W_q = [0.30.1 + 0.40.3, 0.30.2 + 0.40.4] = [0.21, 0.28]

K_love = [0.3, 0.4] @ W_k = [0.30.5 + 0.40.7, 0.30.6 + 0.40.8] = [0.43, 0.5]

V_love = [0.3, 0.4] @ W_v = [0.30.9 + 0.41.1, 0.31.0 + 0.41.2] = [0.61, 0.72]

For 'coding':

Q_coding = [0.5, 0.6] @ W_q = [0.50.1 + 0.60.3, 0.50.2 + 0.60.4] = [0.21, 0.28]

K_coding = [0.5, 0.6] @ W_k = [0.50.5 + 0.60.7, 0.50.6 + 0.60.8] = [0.61, 0.72]

V_coding = [0.5, 0.6] @ W_v = [0.50.9 + 0.61.1, 0.51.0 + 0.61.2] = [0.93, 1.08]

#Step 3: Calculate Attention Scores

Now, calculate the attention scores for each word pair. Attention scores are calculated using the dot product of Q and K, scaled by the square root of the dimension of the vectors (to prevent very large values).

For the word 'I', we calculate the attention scores with 'love' and 'coding'.

Attention Score for 'I' and 'I' = Q_I . K_I = [0.07, 0.1] . [0.19, 0.22] = (0.07 * 0.19) + (0.1 * 0.22) = 0.0133 + 0.022 = 0.0353

Attention Score for 'I' and 'love' = Q_I . K_love = [0.07, 0.1] . [0.43, 0.5] = (0.07 * 0.43) + (0.1 * 0.5) = 0.0301 + 0.05 = 0.0801

Attention Score for 'I' and 'coding' = Q_I . K_coding = [0.07, 0.1] . [0.61, 0.72] = (0.07 * 0.61) + (0.1 * 0.72) = 0.0427 + 0.072 = 0.1147

After calculating these scores, the model would normalize these scores (usually by applying a softmax function) to get attention weights that sum to 1.

#Step 4: Apply Softmax (Normalization)

Using softmax to normalize the attention scores, we can get the final attention weights:

Softmax(Attention Scores):

For 'I', the attention scores are [0.0353, 0.0801, 0.1147]. After applying softmax, we get the weights that will be used to compute the final weighted sum of values (V).

#Step 5: Weighted Sum of Values (V)

Using the attention weights, we compute the final output for 'I' as a weighted sum of the Value vectors:

Final Output for 'I' = weight[0] * V_I + weight[1] * V_love + weight[2] * V_coding.

#Step 6: Predict Next Word

Finally, based on the computed outputs for each word, we can generate a prediction for the next word (in our case, the word most likely to follow).

