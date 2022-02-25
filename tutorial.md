# Introduction

Machine learning, deep learning, neural networks and other magical AI-buzzwords are becoming evermore commonplace. But, let's say you're a curious software developer with limited experience or exposure in this area who is eager to find out more it all. Where and how do you begin to dive in? Is it the right space for you to invest learning in? And is it as complex as _they_ make it out to be?

As a proponent of the [machine](https://cloud.google.com/blog/products/data-analytics/democratization-of-ml-and-ai-with-google-cloud) [learning](https://towardsdatascience.com/democratization-of-ai-de155f0616b5) [democratisation](https://databricks.com/discover/pages/the-democratization-of-artificial-intelligence-and-deep-learning) [movement](https://knowledge.wharton.upenn.edu/article/democratization-ai-means-tech-innovation/), I set about making a relatively simple ‘teaser trailer’ workshop, aimed towards devs curious to just skim the surface of the AI world.

I wrote this workshop with four aims in mind:

1. **Anyone with limited Python experience should be able to participate.** _Why?_ Because if you can code in Python, you can train a basic machine learning model.
2. **Don't overload new folks with a bunch of underlying concepts.** _Why?_ Many machine learning tutorials cover  concepts from the ground up, which can be overwhelming and (in some cases) unnecessary for those wanting to just dip their toes in the ML water. Instead, the focus of this tutorial is on the “_how do I just train a model?_“, and not ”_what are all the underlying the mechanics under-the-hood to train a model?_”.
3. **Keep it fun and engaging.** _Why?_ Learning works best by poking around, changing things up, and making learning customisable to people's own interests. Rather than following a rigid tutorial, I wrote this to be easily customisable so people can put their creativity at work!
4. **It can't be too long — ideally within an hour.** _Why?_ [Because ain't nobody got time for that](https://www.youtube.com/watch?v=N12GtCxwQns).

To test this ‘quick-n-dirty’ workshop idea, I recently put a session together at one of REA Group's Python Dojos, a space where our Python fam get together, share ideas, and learn from each other. At the dojo, I presented [**FastAI**](https://fast.ai/), a wrapper for the more extensive [PyTorch machine learning framework](https://pytorch.org) that helps to abstract away many complexities involved in training new machine learning models.

To make it short and sweet, we covered how to train an image classifier model in just six steps:

1. We [downloaded some training images online](#step-1-download-some-images);
1. We [loaded those images into memory](#step-2-loading-the-images-into-memory);
1. We [trained a baseline (first-pass) model](#step-3-training-the-model);
1. We [figured out what the model learnt](#step-4-what-did-we-learn);
1. We [ran predictions on our baseline model](#step-5-predicting-on-new-images); and,
1. We [made improvements to the baseline model to make it even better](#step-6-improving-our-model).

This post covers each of those six steps in a bit more detail, so that anyone with limited Python experience can train their own image classifier model, be it one to distinguish dog breeds, different art movements, specific ice cream flavours, or even to tell sunflowers from roses!

[**Stop! Disclaimer time.**](https://youtu.be/T_wS0Kw6k9c?t=4) This post is **not** an exhaustive tutorial of all the features of FastAI. It's quick-n-dirty primer, and skips a lot of important details. Folks interested in learning more should check out FastAI's [tutorial](https://docs.fast.ai/tutorial.vision.html) or [free course](https://course.fast.ai)!

For those of you who prefer to dive straight in, the interactive tutorial is fully [**available on GitHub**](https://github.com/alexcu/fastai-primer). And, for those who prefer to watch, the Dojo session given at REA is [available to view on YouTube](https://youtu.be/RcUg8OwQ2ps).

# Step 0: Setup a Google Colab Notebook

As with any workshop, there are _some_ prerequisites. For this tutorial, you'll need a Google account with access to Google Drive in order to run [Google Colaboratory](https://colab.research.google.com).

## Concept: What's a Python Notebook?

_**NB:** If you are familiar with the term already, feel free to skip this section._

If you’ve coded in Python before, you’ve probably just opened a text editor like Visual Studio Code, or PyCharm for Python-specific work. This is great for _engineering_ of code, but **data science isn’t software engineering**—this is where the _science_ comes to play. You have to be able to easily experiment, play around with ideas, and test out hypotheses. Que in Python notebooks.

Python notebooks are a playground of experimentation, akin to [Swift Playgrounds](https://www.apple.com/au/swift/playgrounds/) for any iOS devs out there. They serve as an interactive environment where you can combine coding and code execution as well as marking up rich text (via Markdown or HTML) together in a single ‘document‘.

## What is Google Colaboratory?

Google Colaboratory is a service that provides free (but limited) compute power to run Python notebooks. If you'd like to learn more about notebooks, and specifically _Google Colab notebooks_, you can refer to the [**“Welcome To Colaboratory” notebook**](https://colab.research.google.com/notebooks/intro.ipynb) or the [**“Overview of Colaboratory Features” notebook**](https://colab.research.google.com/notebooks/basic_features_overview.ipynb), both provided by Google.

## Open the FastAI Primer Notebook with a GPU

You can grab a copy of the workshop's ”FastAI Primer” Colab Notebook from [**this GitHub repo**](https://github.com/alexcu/fastai-primer), and then click the **_Open in Colab_** button.

When we perform deep learning tasks (like training neural networks to classify images), [we need a GPU](https://www.run.ai/guides/gpu-deep-learning/) to speed things up. So, make sure your Colab notebook has one to work with:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/GX9QaIN.png?width=518&amp;height=331" alt="Ensure GPU is required" width="518" height="331" />

The second thing we need to do is to set up FastAI in our notebook. Our first code cell looks like the below:

<img style="max-width: 100%;" src="https://i.imgur.com/mNtSerQ.png?width=808&amp;height=192" alt="Code cell" />

To run it, click on the play button, or type <strong><kbd>⌘</kbd>+<kbd>Enter</kbd></strong> when the cell is in focus.

**So what's going on with this code?**

* **Lines 1–3** installs FastAI, upgrades it to the latest version, and also installs [JQ](https://stedolan.github.io/jq/), a lightweight JSON processor which we use later on in the tutorial. The lines prefixed with `!` allow us to interact directly with the shell from within our notebook, which can be pretty handy for running `curl` or `wget` commands in Step 1.
* **Line 4** allows the [matplotlib](https://matplotlib.org) visualisation library to show plots directly in the notebook. This [StackOverflow answer](https://stackoverflow.com/a/61289063) illustrates why.
* **Lines 5 and 6** imports FastAI and then authorises FastAI to interact with your Colab notebook easily. You'll see a warning that will enable the Notebook to interact with Google Drive, however we don't actually store anything to Google Drive.

(_Note, if you can't see line numbers, click on the cog on the righthand side of the code cell, then ensure "Show line numbers" is checked._)


# Step 1: Download some images

Now that our notebook is properly set up, it's time to get some images! **Like instructing a program with code, we ‘instruct’ a machine learning model to learn using all the data it is trained on.** Therefore, getting access to high-quality data and making sure that data is clean is so important!

One way to grab lots of free images online is from the [Creative Commons API](https://api.openverse.engineering/v1/#operation/image_search), recently renamed to the Openverse API.

This is where the fun side comes to play: what would you like to train your classifier on? Below lists a couple of ideas, but you might want to add in something of your own!

<img style="max-width: 100%;" src="https://i.imgur.com/BiFiIcH.png" alt="Ideas to train your model against" />

**Let's break down the code code within Step 1 piece-by-piece.**

## Lines 1–5: Where are we saving our images?

<img style="max-width: 100%;" src="https://i.imgur.com/0Ac8sZ5.png" alt="Setting the root dataset path" />

We have to save all the images we use to train our model _somewhere_. So, we'll download everything to a root `/datasets` directory sitting in the cloud instance running our Google Colab notebook. Each label will have its own subdirectory (e.g., `/datasets/sunflowers`, `/datasets/lavender`, and `/datasets/roses`). We'll store the resulting `Path` in a `datasets_dir` variable.

## Lines 5–9: Starting with a clean slate

<img style="max-width: 100%;" src="https://i.imgur.com/o0GwPmO.png" alt="Empty the datasets directory" />

We'll ensure our root datasets directory is empty by running the `rm` shell command using the `!` prefix. Note that we can access local variables declared in our code cell by referencing them with a `$`, i.e., the `datasets_dir` variable created in the code above.

## Lines 9–18: Setting important variables to download images

<img style="max-width: 100%;" src="https://i.imgur.com/9aZK3aE.png" alt="Customisation of the data we get" />

We now need to set three important variables:

* **How many images we will download from Creative Commons per label?** The more training samples we have, the more things our model will see in training.
* **What will the sizes of those images be?** Higher quality images (in size) will require more training time and resources, but may result in better accuracy.
* **What labels do we actually want to train on?** You can change line 17 to be anything you want, although it should be something easily avaliable on Creative Commons.

## Lines 18–45: Downloading the images

<img style="max-width: 100%;" src="https://i.imgur.com/vpIQI6O.png" alt="Getting the images" />

Here is where we iterate through each of the labels list on line 17 to download `num_imgs_per_label` images. If you open the _Files_ sidebar in Colab, you can see where each of the images were downloaded into under the `/datasets` directory. They are grouped into directory labels, created in lines 21 and 22.

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/x4NgmUG.png?width=362&amp;height=486" alt="Sidebar downloaded images" width="362" height="486" />

The JSON response is downloaded (using cURL) from Creative Commons is parsed via JQ to extract URLs of each image out, which is dumped into the `.txt` files shown above. We then we use wget to download each image at the URLs listed in the `.txt` files, storing the downloaded images in the relevant subdirectories.

We _could_ use cURL or wget to do both, or a Python HTTP library; but remember **we're not engineering code here**. So, just use whatever tool is easier for the job, even if that means mixing shell commands (using the `!` prefix) in with our Python code. (After all, we're experimenting.) For instance, I find cURL to be easier when downloading JSON, but wget easier to download binary content like JPEGs.

## Previewing what we downloaded

<img style="max-width: 100%;" src="https://i.imgur.com/nEAMmzW.png" alt="Preview what we downloaded" />

In the next code cell in Step 1, we have a code cell to preview what images we downloaded. Here, we import FastAI's vision package using a wildcard import. (The FastAI authors make sure discouraged wildcard imports are actually [safe to use](https://stackoverflow.com/a/69791692).)

This vision package comes with some neat helper functions (some which wrap [PyTorch](https://pytorch.org)). For example, here we use:

* [**`get_image_files`**](https://docs.fast.ai/data.transforms.html#get_image_files) to return a list of image files in a directory (in this case, the root `/datasets` directory with each of the three `labels` we set);
* [**`load_image`**](https://docs.fast.ai/vision.core.html#load_image) to load an image into memory, shuffling all images first and only loading the first five;
* [**`show_images`**](https://docs.fast.ai/torch_core.html#show_images) to show multiple images in a grid using [matplotlib](https://matplotlib.org).

This shows in a nice preview as below:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/2dOlOfA.png?width=553&amp;height=398" alt="Output of the preview of our images" width="553" height="398" />

However, as you can see above, some of the images don't look quite right:

* the first impressionism image is a actually a photo of a bus;
* the first cubism image is actually a [post-impressionist artwork](https://en.wikipedia.org/wiki/Portrait_of_Ambroise_Vollard_(Cézanne));
* the second pop art image is a photo of a doll.

Having bad data like this can result in a badly trained model. We'll look at how we can _clean_ this bad data up later, but for now, let's keep the bad data in to spin up a _baseline_ model, just to see how well the model does anyway, without putting in effort to clean the dataset up.

# Step 2: Loading the images into memory

## Lines 1–3: Reserving data for validation

<img style="max-width: 100%;" src="https://i.imgur.com/ADmNGpi.png" alt="Setting validation size" />

**An important part of training a model is to see how well it handles _previously-unseen_ data.**

So, we can split the data we downloaded into two datasets:

  * a **training dataset**, to _teach_ the model what labels are associated with what images;
  * a **validation dataset**, to _validate_ whether the model can _accurately_ assess images it has never seen before (i.e., to assess how well the model predicts correct labels for images _we_ know the ground truth for but _it_ doesn't).

For the purposes of this tutorial, we'll reserve 35% of our all images we downloaded for the validation dataset, leaving the remaining 65% for just training. When we run the cell, you can see just how many images are reserved for both purposes:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/lVQ1wMR.png?width=523&amp;height=194" alt="Reserving data for training" width="523" height="194" />

You can choose to modify the proportion of training versus validation images; by default FastAI uses 20% for validation if not specfifed.

## Lines 4–6: Applying image augmentation: Resize

<img style="max-width: 100%;" src="https://i.imgur.com/p1Ru1rz.png" alt="Resizing images" />

For consistencies sake, it is useful to resize every image to a specific size. Here we will use the FastAI [`Resize` image augmentation transformer](https://docs.fast.ai/vision.augment.html#Resize) (which was imported when we used the wildcard import statement above) to _transform_ every image by resizing them to 256 by 256 pixels. (As a general rule of thumb, the greater the resolution, the greater the accuracy, but this comes with more computation power and therefore training time!)

We could do [additional image augmentation too](https://docs.fast.ai/vision.augment.html), which would generate additional synthetic examples of our images by making rotations, cropping, colour transformations etc. For example, the below images shows 29 augmented examples of a lion from the one source image in the top-right:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/hawOjqh.png?width=343&amp;height=257" alt="Image augmentation example" width="343" height="257" />

For now though, the only augmentation we will apply is resizing images to a consistent size. You may want to try and learn how to weave in additional image augmentation to see how this might improve our baseline model. You can refer to [here](https://docs.fast.ai/vision.augment.html) for more.

## Lines 12–17: Loading our images into a DataLoader

<img style="max-width: 100%;" src="https://i.imgur.com/g2UnReT.png" alt="Loading in the DataLoader" />

As mentioned before, FastAI is a wrapper for PyTorch. It abstracts methods to help load things into PyTorch classes, such as a [`DataLoader`](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html) to enable easy access to each image in our dataset.

FastAI's [**`ImageDataLoaders.from_folder`**](https://docs.fast.ai/vision.data.html#ImageDataLoaders.from_folder) helps us to:

1. find where all our sample images are, i.e., in `/datasets`;
1. define what the vocabulary is in our dataset (i.e., the subrectories given for each of the three labels);
1. set how much data we will reserve for validation; and,
1. apply each of the image transformations to our images (in this case, just resizing).

# Step 3: Training the Model

## Lines 1–3: Defining a Pre-Trained Model

<img style="max-width: 100%;" src="https://i.imgur.com/2byBlyC.png" alt="Set the pretrained model" />

FastAI has [different **pre-trained models**](https://fastai1.fast.ai/vision.models.html#Computer-Vision-models-zoo) which we can piggy back off through a process called [_transfer learning_](https://ruder.io/transfer-learning/). This helps _specialise_ an existing, more generalist model to a specific purpose (in this case, detecting art movements).

The following pre-trained models are all implemented in FastAI ready for us to use:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/zRgTY2B.png?width=561&amp;height=143" alt="Avaliable models for training" width="561" height="143" />

These pre-trained models take a lot of the hard work out for us, so that we don't have to train everything from scratch. For example, [AlexNet](https://en.wikipedia.org/wiki/AlexNet) is a five-layer [convolutional neural network](https://en.wikipedia.org/wiki/Convolutional_neural_network) published in 2012. According to [Neurohive](https://neurohive.io/en/popular-networks/alexnet-imagenet-classification-with-deep-convolutional-neural-networks/):

> AlexNet was trained for 6 days simultaneously on two Nvidia Geforce GTX 580 GPUs on ImageNet, a dataset of over 15 million labelled high-resolution images belonging to roughly 22,000 categories.

Having pre-trained models like this, and specialising them to our own needs using transfer learning, makes training models a lot easier: as the adages go, “don't reinvent the wheel” and ”stand on the shoulders of giants”.

On line 2, we set the `pre_trained_model` variable to `resnet34`, a pre-trained model that was imported during the wildcard import of `fastai.vision.all`. You can change it to any of the other implementations listed in the table embedded within the Colab notebook, each with links to the various papers that first implemented the models.

## Lines 4–6: Setting the number of epochs

<img style="max-width: 100%;" src="https://i.imgur.com/hl4HINb.png" alt="Setting number of epochs" />

During the training process, **the number of times the neural network will work through the entire training dataset is known as an _epoch_**. A singular epoch means that at least every image in the training dataset has the chance to be learnt from.

You could boost this to multiple epochs, meaning that each image within the training dataset will be reviewed more than once, thereby giving the neural network more opportunity to learn from the same data. This will mean that the training process will take longer.

**Training a model is a two-step process**; after each epoch of training, our neural network computes _validation metrics_ on the _validation_ dataset. That is, we test out what our model learnt against a _separate_ (unseen) dataset and test to see if it can accurately predict the correct labels.

## Lines 7–12: Preparing a learner for training

<img style="max-width: 100%;" src="https://i.imgur.com/qURu8yQ.png" alt="Setting up learner" />

FastAI can easily set up a new convolutional neural network (or CNN) to help us learn patterns from our training data. To do this, we call the [`cnn_learner`](https://docs.fast.ai/vision.learner.html#cnn_learner) function, also imported in our wildcard import.

Here we set three key things:

1. We specify the **dataloader** that houses our training and validation datasets in memory.
1. We specify the **pre-trained model** we wish to transfer learn from. In this case, resnet34 from line 2.
1. We specify what **metrics** we wish to report back to us during the training process. In this case, we use `accuracy`, a metric that defines the proportions of correct predictions the CNN is making against our validation dataset once training is done.

This function returns a [`Learner`](https://docs.fast.ai/learner.html#Learner) instance, which is used for our pre-trained model to _learn_ new features according to the vocabulary defined in our dataloader (i.e., the three labels we set). It will also download the pre-trained model from PyTorch's CDN if not already downloaded already.

## Lines 15–16: Fine-tuning the learner

<img style="max-width: 100%;" src="https://i.imgur.com/0Td98oi.png" alt="Fine-tuning the learner" />

When we call the `fine_tune` method on a `Learner`, we're specifying that the underlying pre-trained model should be _fine-tuned_ and specialised to our specific purpose.

**This is the moment of truth!** Executing this code ‘trains‘ the model on our training dataset, and then tests it against our validation dataset. It does this `num_epochs` times, reporting the metrics we specified in line 10.

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/M4NPmeS.png?width=549&amp;height=184" alt="Training in progress" width="549" height="184" />

When training is complete (in this case, in less than about a minute), we can view how our model performed on the validation dataset by interpreting one of our reported metrics. We used accuracy (the proportion of images in the validation dataset that were correctly predicted), which in this case is about 69%.

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/C7DPbEX.png?width=438&amp;height=88" alt="Interpreting accuracy" width="438" height="88" />

You'll also notice `train_loss` and `valid_loss` in the table produced. During the training process, the learner needs a performance measure to help it assess how well the learning process is going. Without diving into the specifics of what this value means, generally the higher the loss, the worse the model's performance is going.

# Step 4: What did we learn?

Now comes the interesting part; we can _interpret_ what the learner correctly and incorrectly learnt. To do this, we can use FastAI's [`ClassificationInterpretation`](https://docs.fast.ai/interpret.html#ClassificationInterpretation) class, which houses a number of tools to help us interpret our model's predictions.

## Lines 1–2: Creating a Classification Interpreter

<img style="max-width: 100%;" src="https://i.imgur.com/qL9ahmL.png" alt="Creating a classification interpreter" />

Firstly, we need to create an interpreter from a given learner using the `from_learner` static method.

## Lines 2–3: Plotting a confusion matrix

<img style="max-width: 100%;" src="https://i.imgur.com/FRYFgW6.png" alt="Plotting a confusion matrix" />

Once we have our interpreter, we can use it to plot a _confusion matrix_. This confusion matrix shows us the disparity between of _correct_ and _incorrect_ predictions (of the labels) made by our model for each image in the validation dataset. When executed, a confusion matrix like the one below shows:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/kL28s5T.png?width=307&amp;height=313" alt="Confusion matrix" width="307" height="313" />

Here, we can see that our model:

* is not very good at distinguishing between cubism and the other two art movements, where the model predicted 23 cubism art images as impressionism, and the other 24 as pop art;
* has learnt certain features about impressionism and pop art, and is able to distinguish between them with only a few mistakes (i.e., 60 and 61 images of impressionism and pop art were, respectively, predicted correctly).

## Lines 6–8: Viewing images with incorrect predictions

<img style="max-width: 100%;" src="https://i.imgur.com/mRgVjNT.png" alt="Viewing the mistakes" />

Diving deeper into the incorrect predictions, we can ask our interpreter to show us which images had the highest loss. (The higher the loss, the worse the performance of the model.) Executing the `plot_top_losses` method on our interpeter shows us:

* images with the highest loss (worst performance), in descending order,
* the predicted label for each image,
* the actual label for each image,
* and the probability of the actual class.

Here's what shows when executed:

<img style="max-width: 100%;" src="https://i.imgur.com/cB84uEF.png" alt="Top 9 images with highest loss" />

It's now pretty clear that we have some **bad data**! For example we've incorrectly included four images of photographs (image 1, 2, 4, and 8) and some of the other images have been incorrectly classified, or do not fit their respective art movements. The model will become confused, trying to learn certain ‘bad‘ features that it shouldn't be learning (e.g., what a photograph looks like instead of what impressionism art looks like in image 1).

As another example, when I ran the FastAI workshop at the Python Dojo, many people who attended and used the dog breeds example noticed that labels downloaded for `Chihuahua` were actually getting images of the [Mexican state of the same name](https://en.wikipedia.org/wiki/Chihuahua_(state)), and not [the dog breed](https://en.wikipedia.org/wiki/Chihuahua_(dog))!

And here comes an important lesson: **a model is only as good as the data you train it.** So, bad training data will inevitably lead to a bad model. In step 6, we'll use FastAI to clean up some of this bad data, but for now, let's explore how our model can be used to run predictions on images it has not yet seen before.

# Step 5: Predicting on new images

Let's run a prediction using our model on the following three random images:

<img style="max-width: 100%;" src="https://i.imgur.com/ppBtbpK.png" alt="Three random images" />

## Lines 1–15: Download random images from the internet with wget

<img style="max-width: 100%;" src="https://i.imgur.com/sJSd6zO.png" alt="Downloading images" />

We'll store URLs to the three images above in a list named `images`, importing the tempfile library to store the downloaded the images in a tempfile. Then, we iterate through each image and can use the `wget` command (via a shell mixin) to download the image into a tempfile (e.g., somewhere inside the `/tmp` directory).


## Lines 16–21: Running a prediction on the learner

<img style="max-width: 100%;" src="https://i.imgur.com/fj7n8fT.png" alt="Running a prediction" />

Once the image is downloaded into the tempfile at `image_path`, we can use the [`predict`](https://docs.fast.ai/learner.html#Learner.predict) method on our learner to make a prediction on the image.

The predict method returns a tuple with three key pieces of information:

<img style="max-width: 100%;" src="https://i.imgur.com/33ZxPIk.png" alt="The tuple" />

These pieces of information refer to the _vocabulary_ within the dataloader produced in Step 2 (i.e., the three labels we used to train the model).

These three pieces of information are:

1. The label the learner is _most_ confident in;
1. The index within the vocabulary of that label corresponds to (as a [`TensorBase`](https://pytorch.org/cppdocs/api/classat_1_1_tensor.html)-typed integer value);
1. The predictions of all three labels in the vocabulary (a `TensorBase` list of floats).

In this case, we can just keep the third element of the tuple and iterate through the data loader's vocabulary (using the `vocab` property on line 20) to see how _confident_ the learner is for each label per image:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/je6zUnS.png?width=500&amp;height=279" alt="How confident the learner is" width="500" height="279" />

1. For the first image of the pug, we can see that the model is most confident that this is pop art; it has learnt the distinguishing colour features that makes pop art different from the other art moves.
1. For the second image of the dog in the suit, there are some mixed confidences; while it thinks the image is also pop art, the confidence of impressionism is also higher than before.
1. Lastly, the image of the sad dog again has a mix of confidences; it's still somewhat confident that this is pop art, but cubism is a close second.

**Our model, therefore, appears to be biased against the pop art label.** It's making many assumptions that most images are  pop art.

While we have a functioning model, it doesn't seem to be that good. In the following step, we'll try looking at cleaning up potentially some bad data that was used to train the model, making our model even better!

# Step 6: Improving our model!

A common theme in training machine learning models is to ensure you have good quality data to train the models on. When we interpreted our model in Step 4, we saw that there was some potentially bad training examples, such as images of photographs or other types of art movements we're not interested in.

Bad data fed into our model leads to a model that learns badly. But, it's also a problem we can fix. In this step, we'll learn how to use FastAI to clean up some of this bad data in a relatively easy manner.

FastAI has a handy data-cleansing widget ([`ImageClassifierCleaner`](https://docs.fast.ai/vision.widgets.html#ImageClassifierCleaner)) that can be used to re-classify bad data into their correct labels, or to prune away bad training examples altogether.

## Creating and executing a cleaner

<img style="max-width: 100%;" src="https://i.imgur.com/uHpvwuD.png" alt="Creating and executing a cleaner" />

When we run the above code, a FastAI widget (an interactive piece of executable code) is imported and executed. These widgets wrap [ipywidgets](https://ipywidgets.readthedocs.io/en/stable/) to make previewing, pruning, and marking bad data a lot easier using HTML forms.

To instantiate the widget, we provide the learner we trained up on line 5 and the maximum number of images we wish to clean at a time. When we execute line 6, the resulting code cell constructs a HTML form that:

* shows images in both the validation and training dataset of each label;
* allows us to flag images for deletion; and,
* allows us to reclassify those images with new labels.

Below shows an example using the widget where we clean up the validation dataset for cubism art.

For each image, we can click the dropdown underneath each image and mark the image as:

* `<keep>` to keep the image labelled as the same;
* `<delete>` to remove the image entirely from the dataset; or,
* select a new label to re-classify the image.

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://media.giphy.com/media/tLxcKWtA1tj6smT2Ce/giphy.gif?width=606&amp;height=187" alt="Example of cleaning" width="606" height="187" />

<!-- https://i.imgur.com/X785kcQ.png -->

Doing this for all images in our dataset is a bit tedious, but it's a good way to get a feel for how the widget works. To speed through the process, click the drop down on the first image, then hit <kbd>TAB</kbd> to cycle to the next image or <kbd>SHIFT</kbd>+<kbd>TAB</kbd> to previous image, and the <kbd>↑</kbd> and <kbd>↓</kbd> keys to move between the dropdown options.

**Important:** Once you have marked a batch of 100 images with the cleaner, move onto the next step below to delete/move the bad data. Then repeat the process for the other training/validation datasets for each label.

**If you don't do this, the changes you have marked will be lost!**

## Deleting and moving bad data

The image cleaner only keeps track of which images we have marked for deletion and which images we have reclassified. **It doesn't actually delete or move the images for you.**

In the second code cell in line 6, we use the `delete()` method on the cleaner to get all cubism training images we marked for deletion, access the `i`th image to delete under the cleaner's `fns` property, and run `rm` to delete the path to that image flagged for deletion:

<img style="max-width: 100%;" src="https://i.imgur.com/NWLEEbL.png" alt="Images marked for deletion" />

This deletes every image flagged as `<delete>` in the above step:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/BvJwfTX.png?width=450&amp;height=237" alt="Deleting bad images" width="450" height="237" />

In the same code cell at line 12, we use the `change()` method on the cleaner to get all the cubism training images we wanted to reclassify with a new label (the `new_label` variable), access the path to the `i`th image using `fns` again, set a new path, and run `mv` to move the image to the new path:

<img style="max-width: 100%;" src="https://i.imgur.com/6fY73Br.png" alt="Images marked for moving" />

This changes the file path of every image flagged with a new label to the new directory:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/Ismh0D5.png?width=588&amp;height=209" alt="Moving bad images" width="588" height="209" />

**⚠️ Before proceeding, make sure you repeat this process for the other datasets and labels.**

## Re-importing cleaned data

Now that we have cleaned up our datasets, the next steps are to:

1. **re-import the data** into our DataLoader (i.e., re-run Step 2);
1. **re-train our model** given the cleaned dataset (i.e., re-run Step 3); and,
1. **re-interpret our learner** to see if cleaning the dataset worked (i.e., re-run Step 4).

For Step 2, re-importing the data will make sure that all data that has been loaded into memory no longer contains images of bad data, or has been reassigned to their correct labels.

When we re-run Step 2, we see that the numbers have reduced:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/QWweCJ9.png?width=535&amp;height=204" alt="Reduced dataset size" width="535" height="204" />

Let's break these new numbers down:

* **impressionism** has been reduced from 132 training images to 100, and to 68 validation images to 58 (total of **158**);
* **cubism art** has been reduced from 129 training images to 66, and to 71 validation images to 39 (total of **105**); and
* **pop art** has been reduced from 129 training images to 87, and to 71 validation images to 38 (total of **125**).

This may introduce _bias_ in our model, we have an uneven dataset, since we have fewer examples of cubism and pop artwork than we do of impressionism artwork. But, when we re-train our model per Step 3, we can see that the model has increased from an overall accuracy of 69% to about **85%**!

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/HF2HADU.png?width=436&amp;height=197" alt="Improved accuracy" width="436" height="197" />

And, if we re-interpret our results, by re-running Step 4, we can see that our confusion matrix has become better:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/JAUtnoE.gif?width=305&amp;height=350" alt="Better confusion matrix" width="305" height="350" />

By cleaning up instances of bad examples, and recategorising them if needed, we have improved the true-positives (actual = predicted) in our validation (i.e., greater numbers along the diagonal top-to-bottom-right). However, we have also introduced _bias_ in our model; the confusion matrix shows much higher true-positive numbers for impressonism   than the other labels. We would need to ensure we have a _balanced_ dataset, so that the model is trained on equal examples for each label.

We can also see that the maximum loss has reduced from **7.54** (from the image in Step 4) to now **4.24**, which is a much better result than the loss we had before cleaning:

<img style="max-width: 100%; display: block; margin-left: auto; margin-right: auto;" src="https://i.imgur.com/d7ea5m9.png?width=553&amp;height=173" alt="Reduced loss" width="553" height="173" />

So, our model _is_ getting better. But we just need to make continued adjustments and re-train as necessary.

# Conclusion

In this tutorial, we covered training a model on a dataset of images that we downloaded from the internet, and then interpreted the results of our trained model. We also saw how to clean up the dataset of bad images, and re-train the model to improve accuracy and performance.

**This primer is by no means exhaustive and extensive**, and skips so many important concepts to machine learning. So, if you find this stuff interesting and want to learn more, I suggest looking at the [free course](https://course.fast.ai) in more detail, or having a look at the FastAI authors' book [_Deep Learning for Coders with Fastai and PyTorch: AI Applications without a PhD_](https://read.amazon.com.au/kp/embed?asin=B08C2KM7NR), which is also avaliable for free on [GitHub](https://github.com/fastai/fastbook).

For the curious, here are some additional exercises you might want to poke at:

* What's the _least_ amount of training images you can use to train a decent model?
* Does increasing/decreasing image quality make a difference?
* Increase the number of epochs. Does it make your model better?
* Modify the training vs. validation dataset split proportions and see the changes in accuracy.
* Switch to a different pre-trained model and compare the results to the previous model you trained.
* How much more cleansing can you do of your datasets to remove bad examples? What's the result to accuracy?
* What happens when you apply different image augmentation to the training images?

Until next time, happy model training! 🤖🎨🧑‍🎨

---

**PS:** I am in no way affiliated with the FastAI, its authors, or Google Colab. I just like teaching people cool things :)
