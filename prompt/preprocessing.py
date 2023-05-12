import pickle
import random
import re
from PIL import Image
import tensorflow as tf
import numpy as np
import collections
from tqdm import tqdm
from encoder import Encoder
import numpy as np


def preprocess_captions(captions, window_size):
    for i, caption in enumerate(captions):
        # Taken from:
        # https://towardsdatascience.com/image-captions-with-attention-in-tensorflow-step-by-step-927dad3569fa
        # Convert the caption to lowercase, and then remove all special characters from it
        caption_nopunct = re.sub(r"[^a-zA-Z0-9]+", ' ', caption[0].lower())
      
        # Split the caption into separate words, and collect all words which are more than 
        # one character and which contain only alphabets (ie. discard words with mixed alpha-numerics)
        clean_words = [word for word in caption_nopunct.split() if ((len(word) > 1) and (word.isalpha()))]
      
        # Join those words into a string
        caption_new = ['<start>'] + clean_words[:window_size-1] + ['<end>']
      
        # Replace the old caption in the captions list with this new cleaned caption
        captions[i] = caption_new
        print(captions)


def get_image_features(image_names, data_folder, vis_subset=100):
    '''
    Method used to extract the features from the images in the dataset using ResNet50
    '''
    image_features = []
    vis_images = []
    resnet = tf.keras.applications.ResNet50(False)  ## Produces Bx7x7x2048
    gap = tf.keras.layers.GlobalAveragePooling2D()  ## Produces Bx2048
    pbar = tqdm(image_names)
    for i, image_name in enumerate(pbar):
        img_path = f'{data_folder}/Images/{image_name}'
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{img_path}' into 2048-D ResNet GAP Vector")
        with Image.open(img_path) as img:
            img_array = np.array(img.resize((224,224)))
            # reduce to 3 channels
            img_array = img_array[:,:,:3]
        img_in = tf.keras.applications.resnet50.preprocess_input(img_array)[np.newaxis, :]
        image_features += [gap(resnet(img_in))]
        if i < vis_subset:
            vis_images += [img_array]
    return image_features, vis_images

#    train_prompts_features = get_prompt_features(train_image_names, train_prompts)
def get_prompt_features(image_names, prompts):
    prompt_features = []
    pbar = tqdm(image_names)
    for i, prompt in enumerate(prompts):
        embedding = tf.keras.layers.Embedding(len(prompt), 128)
        pbar.set_description(f"[({i+1}/{len(image_names)})] Processing '{prompt}' into 2048-D ResNet GAP Vector")
        prompt_features += [embedding(np.array(prompt))]
    return prompt_features


def load_data(data_folder):
    text_file_path = f'{data_folder}/response.txt'

    with open(text_file_path) as file:
        examples = file.read().splitlines()[1:]
    
    #map each image name to a list containing its prompt
    image_names_to_reponses = {}
    image_names_to_prompts = {}
    for example in examples:
        img_name, caption, prompt = example.split(',', 2)
        image_names_to_reponses[img_name] = [caption]
        image_names_to_prompts[img_name] = [prompt]

    #randomly split examples into training and testing sets
    shuffled_images = list(image_names_to_reponses.keys())
    random.seed(0)
    random.shuffle(shuffled_images)
    #need to change [:1] and [1:] to [:100] and [100:] for full dataset
    test_image_names = shuffled_images[:1]
    train_image_names = shuffled_images[1:]

    def get_all_responses(image_names):
        to_return = []
        for image in image_names:
            caption = image_names_to_reponses[image]
            to_return.append(caption)
        return to_return

    def get_all_prompts(image_names):
        to_return = []
        for image in image_names:
            prompt = image_names_to_prompts[image]
            to_return.append(prompt)
        return to_return


    # get lists of all the captions in the train and testing set
    train_responses = get_all_responses(train_image_names)
    test_responses = get_all_responses(test_image_names)
    train_prompts = get_all_prompts(train_image_names)
    test_prompts = get_all_prompts(test_image_names)
    

    #remove special charachters and other nessesary preprocessing
    window_size = 20
    preprocess_captions(train_responses, window_size)
    preprocess_captions(test_responses, window_size)
    preprocess_captions(train_prompts, window_size)
    preprocess_captions(test_prompts, window_size)

    # count word frequencies and replace rare words with '<unk>'
    word_count = collections.Counter()
    for caption in train_responses:
        word_count.update(caption)

    def unk_captions(captions, minimum_frequency):
        for caption in captions:
            for index, word in enumerate(caption):
                if word_count[word] <= minimum_frequency:
                    caption[index] = '<unk>'

    unk_captions(train_responses, 50)
    unk_captions(test_responses, 50)
    unk_captions(train_prompts, 50)
    unk_captions(test_prompts, 50)

    # pad captions so they all have equal length
    def pad_captions(captions, window_size):
        for caption in captions:
            caption += (window_size + 1 - len(caption)) * ['<pad>'] 
    
    pad_captions(train_responses, window_size)
    pad_captions(test_responses,  window_size)
    pad_captions(train_prompts, window_size)
    pad_captions(test_prompts,  window_size)

    # assign unique ids to every work left in the vocabulary
    word2idx = {}
    vocab_size = 0
    for caption in train_responses:
        for index, word in enumerate(caption):
            if word in word2idx:
                caption[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                caption[index] = vocab_size
                vocab_size += 1
    for caption in test_responses:
        for index, word in enumerate(caption):
            caption[index] = word2idx[word] 
    
    for prompt in train_prompts:
        for index, word in enumerate(prompt):
            if word in word2idx:
                prompt[index] = word2idx[word]
            else:
                word2idx[word] = vocab_size
                prompt[index] = vocab_size
                vocab_size += 1
    
    for prompt in test_prompts:
        for index, word in enumerate(prompt):
            prompt[index] = word2idx[word]

    # use ResNet50 to extract image features
    print("Getting training embeddings")
    train_image_features, train_images = get_image_features(train_image_names, data_folder)
    train_prompts_features = get_prompt_features(train_image_names, train_prompts)
    train_prompts_features = tf.reshape(train_prompts_features, (1,-1))
    print(train_image_features)
    print("Getting testing embeddings")
    test_prompts_features = get_prompt_features(test_image_names, test_prompts)
    test_prompts_features = tf.reshape(test_prompts_features, (1,-1))
    test_image_features,  test_images  = get_image_features(test_image_names, data_folder)


    return dict(
        train_captions          = np.array(train_responses),
        test_captions           = np.array(test_responses),
        train_image_features    = np.array(train_image_features),
        test_image_features     = np.array(test_image_features),
        train_images            = np.array(train_images),
        test_images             = np.array(test_images),
        word2idx                = word2idx,
        idx2word                = {v:k for k,v in word2idx.items()},
    )


def create_pickle(data_folder):
    with open(f'{data_folder}/data.p', 'wb') as pickle_file:
        pickle.dump(load_data(data_folder), pickle_file)
    print(f'Data has been dumped into {data_folder}/data.p!')


if __name__ == '__main__':
    ## Download this and put the Images and captions.txt indo your ../data directory
    ## Flickr 8k Dataset: https://www.kaggle.com/datasets/adityajn105/flickr8k?resource=download
    data_folder = '/Users/kevin/Desktop/cs1470/RoadMaster_GPT/prompt/data'
    create_pickle(data_folder)