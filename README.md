# Medico: Transparency in Medical Image Segmentation

<!-- # please respect the structure below-->
*See the [MediaEval 2021 webpage](https://multimediaeval.github.io/editions/2021/) for information on how to register and participate.*

#### Task Description
The fight against colorectal cancer requires better diagnosis tools. Computer-aided diagnosis systems can reduce the chance that diagnosticians overlook a polyp during a colonoscopy. As machine learning becomes more common, even in high-risk fields like medicine, the need for transparent systems becomes more critical. In this case, transparency is defined as giving as much detail as possible on the different parts that make up a machine learning pipeline, including everything from data collection to final prediction. This task focuses on robust, transparent, and efficient algorithms for polyp segmentation.

Medical image segmentation is a topic that has garnered a lot of attention over the last few years. Compared to classification and object detection, segmentation gives a more precise region of interest for a given class. This is immensely useful for the doctors as it not only specifies that an image contains something interesting but also where to look at which also provides some kind of inherent explanation. Colonoscopies are a perfect use-case for medical image segmentation as they contain a great variety of different findings that may be easily overlooked during the procedure. Furthermore, transparent and interpretable machine learning systems are important to explain the *whys* and the *hows* of the predictions. This is especially important in medicine, where conclusions based on wrong decisions resulted from either biased or incorrect data, faulty evaluation or simply a bad model could be fatal. For this reason, the *Medico: Transparency in Medical Image Segmentation* task aims to develop automatic segmentation systems that are transparent and explainable.

The data consists of a large number of endoscopic images of the colon, which have been labeled by expert gastroenterologists.

*Subtask 1: Polyp Segmentation:* The polyp segmentation task asks participants to develop algorithms for segmenting polyps in images taken from endoscopies. The main focus of this task is to achieve high segmentation metrics on the supplied test dataset. Since [Medico 2020](https://multimediaeval.github.io/editions/2020/tasks/medico/), we have extended the development dataset and created a new testing dataset to which the submissions will be evaluated on.

*Subtask 2: Algorithm Efficiency* The algorithm efficiency task is similar to subtask one, but puts a stronger emphasis on the algorithm's speed in terms of frames-per-second. To ensure a fair evaluation, this task requires participants to submit a Docker image so that all algorithms are evaluated on the same hardware.

*Subtask 3: Transparent Machine Learning Systems* The transparency task tries to measure the transparency of the systems used for the aforementioned segmentation tasks. The main focus for this task is to evaluate systems from a transparency point of view, meaning for example explanations of how the model was trained, the data that was used, and interpretation of a model's predictions.

Participants are encouraged to make their code public with their submission.

#### Task Schedule

| | | 
| :---  | :---  |
| 1 July, 2021 | Development data release | 
| 1 October, 2021 | Test data release | 
| 1 November, 2021 | Runs due | 
| 22 November, 2021 | Working notes due |
| 6-8 December, 2021 | MediaEval 2021 Workshop |
| | | 

#### Development Dataset (Released)

The development dataset consists of 1,360 images of polyps with corresponding segmentation masks. Note that the dataset is based on HyperKvasr, a large public dataset contrainig diverse visual content from the gastrointestinal tract. Below you will find a link to the development dataset and HyperKvasir.

* [Development Dataset](https://drive.google.com/drive/folders/16MdULl8bNX3wp0OzjU33BV6EJ_YScyGd?usp=sharing)
* [HyperKvasir](https://datasets.simula.no/hyper-kvasir/)

#### Test Datset (Coming Soon)
The testing dataset will be released on October 1st.

#### Task organizers
* Steven Hicks, SimulaMet, Norway steven (at) simula.no
* Debesh Jha, SimulaMet, Norway  debesh (at) simula.no
* Vajira Thambawita, SimulaMet and OsloMet, Norway 
* Thomas de Lange, Bærum Hospital, Norway
* Sravanthi Parasa, Swedish Medical Center, Sweden
* Michael Riegler, SimulaMet, Norway  
* Pål Halvorsen, SimulaMet and OsloMet, Norway 

