# Medico: Transparency in Medical Image Segmentation

*See the [MediaEval 2021 webpage](https://multimediaeval.github.io/editions/2021/) for information on how to register and participate.*

*See the [Medico task overview paper]() for more details.*

### Task Description
Medical image segmentation is a topic that has garnered a lot of attention over the last few years. Compared to classification and object detection, segmentation gives a more precise region of interest for a given class. This is immensely useful for the doctors as it not only specifies that an image contains something interesting but also where to look at which also provides some kind of inherent explanation. Colonoscopies are a perfect use-case for medical image segmentation as they contain a great variety of different findings that may be easily overlooked during the procedure. Furthermore, transparent and interpretable machine learning systems are important to explain the *whys* and the *hows* of the predictions. This is especially important in medicine, where conclusions based on wrong decisions resulted from either biased or incorrect data, faulty evaluation or simply a bad model could be fatal. For this reason, the *Medico: Transparency in Medical Image Segmentation* task aims to develop automatic segmentation systems that are transparent and explainable.

### Subtasks and Submission
The 2021 edition of the Medico Multimedia Task provides threedifferent subtasks, namelythe polyp segmentation task,the efficientsegmentation task, andthe transparent machine learning systemstask. Only the polyp segmentation task is required to participate.Each task allows for a total of five submissions each. Submissions can be sendt to steven@simula.no.

#### Subtask 1: Polyp Segmentation
The polyp segmentation task targets high-performing polyp segmentation systems. Using the provided development dataset, participants are asked to develop models that automatically segment the presence of colon polyps in a given image. The main focus of this task is to achieve high segmentation metrics on the supplied testing dataset. Submission to this task should be a zip file containing a predicted segmentation mask using the .png file format for each image in the testing dataset. Each predicted mask should use the same resolution as the input image and have the same filename.

Submissions will be evaluated based on the precision of the predicted masks using various segmentation metrics like pixel accuracy, precision, recall, Sørensen–Dice coefficient (Dice), and Intersection over Union (IoU). The primary metric used to rank the submissions will be IoU. The participants will receive a .csv file containing the evaluation metrics for each run.

#### Subtask 2: Algorithm Efficiency
The efficient segmentation task aims for efficient segmentation systems while still obtaining a satisfactory prediction accuracy. Model efficiency is measured in the number of frames that a model can process per second. The motivation behind this is the need for real-time detection systems that can be used during a live endoscopy procedure. In addition to using the development dataset to develop a polyp segmentation model, this task also requires the participants to submit a Docker image of their implementation to be evaluated on the organizers' hardware. The Docker submission should generate a .csv submission file that contains the name of the segmented image and the time (in seconds) used to perform the segmentation.

Models will be evaluated based on the performance metrics used to evaluate the polyp segmentation task and the number of frames that can be segmented per second. Submission will be ranked based on a balanced metric between predictive performance and speed. All submissions are evaluated on what can be considered consumer-grade hardware, that is, a computer running Arch Linux with an Intel Core i9-10900K processor, an Nvidia GeForce RTX 3090 graphics processing unit (GPU), and 32 gigabytes of RAM.

#### Subtask 3: Transparent Machine Learning Systems
The goal of the transparent machine learning system task} is to promote more transparency in medical applications of machine learning. The motivation behind this task is rooted in a general lack of transparency in medical machine learning research. A lot of work is often published using private data, closed-source implementations, and lackluster evaluations, making the systems not very reproducible. We leave it to the participants to determine what makes a machine learning system transparent. Still, some ideas include failure analysis, ablation studies, model explanations, open and commented source code, and detailed implementation descriptions.

Submissions to this task will be evaluated by a committee comprised of computer scientists and expert gastroenterologists. The committee will evaluate the submissions from different perspectives. For example, the medical doctors will look at the system from a clinic point of view, assessing transparency based on how it can be used in the clinic. The computer scientists will look at the technical transparency of the submissions, like source code descriptions and the clarity of the implementation. Each team that submits to this task will receive a report on the level of transparency determined by the evaluation committee. 

### Task Schedule

| | | 
| :---  | :---  |
| 1 July, 2021 | Development data release | 
| 1 October, 2021 | Test data release | 
| 1 November, 2021 | Runs due | 
| 22 November, 2021 | Working notes due |
| 6-8 December, 2021 | MediaEval 2021 Workshop |
| | | 

### Development Dataset (Released)
The development dataset consists of 1,360 images of polyps with corresponding segmentation masks. Note that the dataset is based on HyperKvasr, a large public dataset contrainig diverse visual content from the gastrointestinal tract. Below you will find a link to the development dataset.

* [Development Dataset (Google Drive)](https://drive.google.com/drive/folders/16MdULl8bNX3wp0OzjU33BV6EJ_YScyGd?usp=sharing)

### Test Dataset (Coming Soon)
The testing dataset will be released on October 1st.

### Task Organizers
* Steven Hicks, SimulaMet, Norway steven (at) simula.no
* Debesh Jha, SimulaMet, Norway  debesh (at) simula.no
* Vajira Thambawita, SimulaMet and OsloMet, Norway 
* Hugo Hammer, OsloMet, Norway 
* Thomas de Lange, Bærum Hospital, Norway
* Sravanthi Parasa, Swedish Medical Center, Sweden
* Michael Riegler, SimulaMet, Norway  
* Pål Halvorsen, SimulaMet and OsloMet, Norway 

