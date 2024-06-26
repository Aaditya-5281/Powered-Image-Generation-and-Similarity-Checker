import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, cifar10, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.metrics import normalized_mutual_info_score
from skimage.metrics import structural_similarity as ssim
import os
from google.colab import files

# Set up folders
UPLOAD_FOLDER = '/content/uploads'
GAN_IMAGES_FOLDER = '/content/gan_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GAN_IMAGES_FOLDER, exist_ok=True)

def load_data(dataset='cifar100'):
    if dataset == 'mnist':
        (X_train, _), (_, _) = mnist.load_data()
        X_train = X_train / 127.5 - 1.0
        X_train = np.expand_dims(X_train, axis=-1)
    elif dataset == 'cifar10':
        (X_train, _), (_, _) = cifar10.load_data()
        X_train = X_train / 127.5 - 1.0
    elif dataset == 'cifar100':
        (X_train, _), (_, _) = cifar100.load_data()
        X_train = X_train / 127.5 - 1.0
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    return X_train

def build_generator(img_shape):
    model = Sequential()
    model.add(Dense(256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))
    return model

def build_discriminator(img_shape):
    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def save_imgs(epoch, generator, img_shape, dataset):
    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images 0 - 1

    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    count = 0
    for i in range(5):
        for j in range(5):
            if img_shape[-1] == 1:
                axs[i, j].imshow(gen_imgs[count].reshape(img_shape[:-1]), cmap='gray')
            else:
                axs[i, j].imshow(gen_imgs[count])
            axs[i, j].axis('off')
            count += 1
    plt.tight_layout()
    plt.savefig(f"{GAN_IMAGES_FOLDER}/{dataset}_{epoch}.png")
    plt.close()

def train_gan(dataset, epochs=50, batch_size=128):
    X_train = load_data(dataset)
    img_shape = X_train.shape[1:]

    generator = build_generator(img_shape)
    discriminator = build_discriminator(img_shape)

    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

    z = Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, 100))
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % 5 == 0:
            # Calculate accuracy
            _, accuracy_real = discriminator.evaluate(imgs, valid, verbose=0)
            _, accuracy_fake = discriminator.evaluate(gen_imgs, fake, verbose=0)
            accuracy = 0.5 * (accuracy_real + accuracy_fake)

            print(f"Epoch {epoch}, D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}, Accuracy: {accuracy*100:.2f}%")
            save_imgs(epoch, generator, img_shape, dataset)

    return generator

# Choose a dataset
dataset = 'cifar100'  # You can change this to 'cifar10' or 'cifar100'

# Train the GAN
generator = train_gan(dataset, epochs=100, batch_size=128)

def check_similarity(uploaded_image_path, dataset):
    if dataset == 'mnist':
        target_size = (28, 28)
        color_mode = 'grayscale'
    else:
        target_size = (32, 32)
        color_mode = 'rgb'

    uploaded_img = image.load_img(uploaded_image_path, target_size=target_size, color_mode=color_mode)
    uploaded_img = image.img_to_array(uploaded_img)
    uploaded_img = uploaded_img / 255.0  # Normalize to [0, 1]

    # Load generated images
    gen_images = []
    for file in os.listdir(GAN_IMAGES_FOLDER):
        if file.startswith(f"{dataset}_") and file.endswith(".png"):
            img_path = os.path.join(GAN_IMAGES_FOLDER, file)
            img = image.load_img(img_path, target_size=target_size, color_mode=color_mode)
            img = image.img_to_array(img)
            img = img / 255.0  # Normalize to [0, 1]
            gen_images.append(img)

    if not gen_images:
        return "No generated images found to compare."

    # Compute similarity
    max_similarity = 0
    for gen_img in gen_images:
        if uploaded_img.shape != gen_img.shape:
            continue

        if color_mode == 'grayscale':
            similarity = ssim(uploaded_img[:,:,0], gen_img[:,:,0])
        else:
            similarity = ssim(uploaded_img, gen_img, multichannel=True)

        max_similarity = max(max_similarity, similarity)

    # Define a threshold for similarity
    threshold = 0.5

    if max_similarity > threshold:
        return f"Image is similar to generated images. Similarity score: {max_similarity:.4f}"
    else:
        return f"Image is not similar to generated images. Similarity score: {max_similarity:.4f}"

# Upload an image
uploaded = files.upload()

if uploaded:
    file_path = next(iter(uploaded))
    full_path = os.path.join(UPLOAD_FOLDER, file_path)
    with open(full_path, 'wb') as f:
        f.write(uploaded[file_path])

    # Display the uploaded image
    plt.imshow(plt.imread(full_path))
    plt.axis('off')
    plt.show()

    # Check similarity
    result = check_similarity(full_path, dataset)
    print(result)

    # Display a sample of generated images for comparison
    gen_samples = [f for f in os.listdir(GAN_IMAGES_FOLDER) if f.startswith(f"{dataset}_")]
    if gen_samples:
        sample = np.random.choice(gen_samples)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(plt.imread(full_path))
        plt.title("Uploaded Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(plt.imread(os.path.join(GAN_IMAGES_FOLDER, sample)))
        plt.title("Sample Generated Image")
        plt.axis('off')
        plt.show()
