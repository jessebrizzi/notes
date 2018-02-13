- Seems like every big tech company is coming out with their own open-source GPU DL library
- ML is still very much in its early days
  - Still figuring things out when compared to general software engineering like source control and continuous builds.
  - A lot of best practices that not everyone is using, and a lot of things are still done ad-hoc


Useful model converter https://github.com/ysh329/deep-learning-model-convertor

dl4j lib comparison https://deeplearning4j.org/compare-dl4j-tensorflow-pytorch

Imperative vs Symbolic paradigms
----------------------------------
- Among other popular deep learning libraries, Torch, Chainer, and Minerva embrace the imperative style.
- Examples of symbolic-style deep learning libraries include Theano, CGT, and TensorFlow.
- We might also view libraries like CXXNet and Caffe, which rely on configuration files, as symbolic-style libraries.
  - In this interpretation, we’d consider the content of the configuration file as defining the computation graph.
- Imperative programs tend to be more flexible
  - Imperative programs are more native than symbolic programs.
  - It’s easier to use native language features.
  - For example, it’s straightforward to print out the values in the middle of computation or to use native control flow and loops at any point in the flow of computation.
- Symbolic Programs Tend to be More Efficient
  - Both in terms of memory and speed
  - Symbolic programs can safely reuse the memory for in-place computation
    - but if we unexpectedly need results from earlier nodes in the graph, we can not get them.
    - especially important when computing the backwards pass through the graph
  - Symbolic programs can also perform operation folding optimizations
    - Combining operations to run as a single kernel on the GPU where possible
    - Operation folding improves computation efficiency.
  - Symbolic models are easier to manage in terms of loading and saving since the definition of the graph and the execution of the graph are separate.
- Easier to write parameter updates in imperative programs, especially when you have multiple updates that refer to each other.
  - most symbolic deep learning libraries fall back on the imperative approach to perform updates, while using the symbolic approach to perform gradient calculation.

- Static Computation Graphing (Symbolic Paradigm)
  - define the computation graph once, and uses a session to execute ops in the graph many times.
  - Can optimized the graph at the start
  - Good for fixed size Net (feed-forward, CNNS)
- Dynamic Computation Graphing (Imperative Programming)
  - Are build and rebuilt at runtime which lets you use standard python statements.
  - At run time the system generation the graph structure
  - Useful for when the graph structure needs to change at run time, like in RNN's
  - Makes debugging easy since an error is not thrown in a single call to execute the graph after its compiled, but at the specific line in the dynamic graph at run time.



Important factors
------------------
- Academia vs Industry
- Community support
  - Pretrained models
  - Research paper repos

- Codebase Quality
  - Is the code actively maintained?

- Train to Production pipeline
  - Train in a fast to prototype language (python) and deploy in your production language (java/scala, c++, whatever)
  - train locally if you have the hardware vs training on cloud services

- Performance
  - Benchmarks (oldish) https://arxiv.org/pdf/1608.07249.pdf
  - In general, the performance does not scale very well on many-core CPUs. In many cases, the performance of using 16 CPU cores is only slightly better than that of using 4 or 8 CPU cores. Using CPU computing platform, TensorFlow has a relatively better scalability compared with other tools.
  - With a single GPU platform, Caffe, CNTK and Torch perform better than MXNet and TensorFlow on FCNs; MXNet is outstanding in CNNs, especially the larger size of networks, while Caffe and CNTK also achieve good performance on smaller CNN; For RNN of LSTM, CNTK obtains excellent time efficiency, which is up to 5-10 times better than other tools.
  - With the parallelization of data during training, all the multi-GPU versions have a considerable higher throughput and the convergent speed is also accelerated. CNTK performs better scaling on FCN and AlexNet, while MXNet and Torch are outstanding in scaling CNNs.
  - GPU platform has a much better efficiency than many-core CPUs. All tools can achieve significant speedup by using contemporary GPUs.
  - To some extent, the performance is also affected by the design of configuration files. For example, CNTK allows the end users to fine-tune the system and trade off GPU memory for better computing efficiency, and MXNet gives the users to configure the auto-tune setting in using cuDNN library.
  - Scalability across multiple GPUs
    - MXNet and Torch scale the best and TensorFlow scales the worst
  - Overall the performance of TensorFlow is lacking compared to the other tools

- The ability to Scale
  - In both training and production
- Development speed
  - The barriers for entry to get started (learning, documentation, programming languages)
- Portability
  - ability to run on different platforms ranging from mobile phones to massive server farms



mxnet
-------
- https://github.com/apache/incubator-mxnet
- Python, Scala, R, Julia, C++, Perl, Go, Javascript, matlab (largest # of languages officially supported)
- Apache, Amazon, University of Washington and Carnegie Mellon University
- keras support as backed https://github.com/dmlc/keras
- Watches 1109, stars, 12605, Forks 4644, Median Issue resolution 53 days, Percent Open Issues 11%
- https://mxnet.apache.org/
- Research Citations - 246
  - https://scholar.google.com/scholar?cites=3990509978676884239&as_sdt=40005&sciodt=0,10&hl=en
  - Mxnet: A flexible and efficient machine learning library for heterogeneous distributed systems
  - Tianqi Chen, Mu Li, Yutian Li, Min Lin, Naiyan Wang, Minjie Wang, Tianjun Xiao, Bing Xu, Chiyuan Zhang, Zheng Zhang
  - 2015
- Model zoo
  - https://mxnet.apache.org/model_zoo/index.html
  - Covers most of the expected pretrained models you would need.


- Static Computation Graphing
- Scaling Performance
  - Amazon has found that you can get up to an 85% scaling efficiency with Mxnet
    - http://www.allthingsdistributed.com/2016/11/mxnet-default-framework-deep-learning-aws.html
    - part of the reason they have thrown their weight behind it.
- Mobile Support
  - Has it with examples of running the single file C++ version of MXNet on IOS and Android
  - Converter to CoreML for iOS
  - Other Libs have better support
- Browser Support
  - https://github.com/rupeshs/machineye
  - http://rupeshs.github.io/machineye/


- Gluon
  - https://mxnet.incubator.apache.org/gluon/index.html
  - Gluon library in Apache MXNet provides a clear, concise, and simple API for deep learning.
  - Colaberation between AWS and Microsoft
  - Simple, Easy-to-Understand Code: Gluon offers a full set of plug-and-play neural network building blocks, including predefined layers, optimizers, and initializers.
  - Flexible, Imperative Structure: Gluon does not require the neural network model to be rigidly defined, but rather brings the training algorithm and model closer together to provide flexibility in the development process.
  - Dynamic Graphs: Gluon enables developers to define neural network models that are dynamic, meaning they can be built on the fly, with any structure, and using any of Python’s native control flow.
  - High Performance: Gluon provides all of the above benefits without impacting the training speed that the underlying engine provides.
  - Has its own model Zoo
    - https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html

- mxnet model server
  - https://github.com/awslabs/mxnet-model-server
  - https://aws.amazon.com/blogs/ai/introducing-model-server-for-apache-mxnet/
  - AWS's attempt at a model serving framework for MXNet

- https://techburst.io/mxnet-the-real-world-deep-learning-framework-2690e56ef81f



caffe
-------
- https://github.com/BVLC/caffe
- C++, python, Matlab
- UC Berkeley
- Watches 2127, stars 22,028, Forks 13513, Ave Issue Resolution 61 Days, Percent Open Issues 15%
- Research Citations - 6275
  - https://scholar.google.com/scholar?cites=1739257544589912763&as_sdt=40005&sciodt=0,10&hl=en
  - Caffe: Convolutional Architecture for Fast Feature Embedding
  - Yangqing Jia, Evan Shelhamer, Jeff Donahue, Sergey Karayev,  Jonathan Long, Ross Girshick, Sergio Guadarrama, Trevor Darrell
  - 2014
- Model zoo
  - https://github.com/BVLC/caffe/wiki/Model-Zoo
  - One of the firsts, if not the first, model zoo.
  - Very large, and has links to many specialized pretrained nets  apart from the typical dataset you see trained on.
  - Lost of models from publish research papers.


- The original Caffe framework was useful for large-scale product use cases, especially with its unparalleled performance and well tested C++ codebase.
- Caffe has some design choices that are inherited from its original use case: conventional CNN applications.
- As new computation patterns have emerged, especially distributed computation, mobile, reduced precision computation, and more non-vision use cases, its design has shown some limitations.

- Basically the first mainstream production grade DL library
- last remaining relevant Academia built DL framework
- Not very flexible
- Able to train a net from your data without writing any code using binaries and config files.
  - These configuration files are very cumbersome
  - The prototxt for ResNet-152 is 6775 lines long.
- Good for production situations with pure c++ implementation (deploy without python) and speed
- Good for feedforward networks, image processing, and for fine-tuning pretrained nets
- Not good for recurrent networks
- Does not support Auto differentiation
- In caffe, the graph is treated as a collection of layers, as opposed to nodes of single tensor operations
  - A layer is a composition of multiple tensor operations
  - If you want to develop new layer types you have to define the full forward and backwards gradient updates
  - layers are building blocks for the net
  - They are not very flexible and there are a lot of them that duplicate a lot of similar logic internally.
- Very verbose in layer and network definitions
  - Have to define CPU and GPU layer code separately.
  - Most models are statically defined in plaintext and not programatically.
- Main developer now works on TensorFlow
- Being first to market means that a lot of early research and models where written with caffe, and the research that built off of that forked and continued to use the same code base. So you will find a lot of state of the art work to this day still using caffe despite its limitations.



TensorFlow
------------
- https://github.com/tensorflow/tensorflow
- Python, C++, Java, Go (the APIs in languages other than Python are not yet covered by the API stability promises)
- Community Support for C#, Haskell, Julia, Ruby, Rust, and Scala
- Google
  - Attempt to build a single DL framework for everything DL related (research, production, all of the DL paradigms)
- Keras support as backed
  - In fact has an optimized version built in to the library
- Watches 7164, Stars 84259, Forks 41,117, Median Issue Resolution 8 days, Percent Open Issues 16%
- research citations - 2298
  - https://scholar.google.com/scholar?cites=4870469586968585222&as_sdt=40005&sciodt=0,10&hl=en
  - TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems
  - Martín Abadi, Ashish Agarwal, Paul Barham, Eugene Brevdo, Zhifeng Chen, Craig Citro, Greg S. Corrado, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Ian Goodfellow, Andrew Harp, Geoffrey Irving, Michael Isard, Yangqing Jia, Rafal Jozefowicz, Lukasz Kaiser, Manjunath Kudlur, Josh Levenberg, Dan Mane, Rajat Monga, Sherry Moore, Derek Murray, Chris Olah, Mike Schuster, Jonathon Shlens, Benoit Steiner, Ilya Sutskever, Kunal Talwar, Paul Tucker, Vincent Vanhoucke, Vijay Vasudevan, Fernanda Viegas, Oriol Vinyals, Pete Warden, Martin Wattenberg, Martin Wicke, Yuan Yu, Xiaoqiang Zheng
  - 2016
- Model Zoo
  - https://github.com/tensorflow/models

- Made to use symbolic programing
  - clear separation between defining the computation graph, and compiling it. Computation happens at last step.
  - More efficient due to safe value reusability
- Static Computation Graphing
- Great community size
- Some of the best documentation on the web for a DL library
- One of the best for production
  - Built with distributed computing in mind
- SKFlow ???
  - Implements a lot of SKLearn
- Mobile deployment
  - TensorFlow Mobile and TensorFlow Light
    - Use TF Mobile for production
  - Support for both Android and IOS

- TensorBoard
  - Data visualization tool that comes packaged with TensorFlow
  - Was created to help you understand the flow of tensors through your model for debugging and optimization
  - You can create summary operations in your graph that take tensor inputs and give tensor outputs that TensorBoard can read for display and analysis.

- TensorFlow Serving
  - https://github.com/tensorflow/serving
  - A flexible, high-performance serving system for ML models, designed for production environments.
  - Serving is how you apply a ML model after you train it
  - C++ libraries, binaries, and docker/k8 containers
  - hosted service on Google CloudML
  - Can be used as an RPC server or a set of libraries
  - goals
    - Online, low latency
      - Optimizes for throughput
      - mini batch scheduler to schedule threads between multiple models
    - multiple models in a single process
    - multiple version of a model loaded over time
    - compute cost varies in real-time to meet product time
      - auto scaling in the CloudML, Docker & K8s
    - Aim for the efficiency of mini-batching at training time
      - except with requests arriving asynchronously
  - API interfaces
    - Complete
      - Prediction
    - Comping-soon (NEED TO CHECK)
      - Regression
      - Classification
      - multiInference

- TensorFlow Fold
  - Adds Dynamic graph functionality to TensorFlow
  - https://github.com/tensorflow/fold



CNTK
-----
- https://github.com/Microsoft/CNTK
- Python, C#, C++, R
- Microsoft
- Keras support as backed
- Watches 1318, Stars 13489, Forks 3525, Median Issue Resolution Time 21 days, Percent Open issues 12%
- Research Citations - 12
  - https://scholar.google.com/scholar?cites=14941870274579355971&as_sdt=40005&sciodt=0,10&hl=en
  - Microsoft's Open-Source Deep-Learning Toolkit
  - Frank Seide, Amit Agarwal
  - 2016

http://images.nvidia.com/events/sc15/pdfs/CNTK-Overview-SC150-Kamanev.pdf
- CNTK: Computational Network ToolKit
  - Created by MSR Speech researchers several years ago
- Unified framework for building:
  - Deep Neural Networks (DNNs)
  - Recurrent Neural Networks (RNNs)
  - Long Short Term Memory networks (LSTMs)
  - Convolutional Neural Networks (CNNs)
  - Deep Structured Semantic Models (DSSMs)
  - and few other things…
- All types of deep learning applications: speech, vision and text

- Open source
  - Currently hosted on CodePlex, GitHub migration to be done soon
  - Contributors from Microsoft and external (MIT, Stanford etc)
- Runs on Linux and Windows
  - Project Philly runs 100% on Linux
- Efficient GPU and CPU implementations
- GPU implementation uses highly-optimized libraries from NVIDIA:
  - CUB
  - cuDNN
  - and of course other things like cuBLAS, cuSPARSE, cuRAND etc.

- Distributed training
  - Can scale to hundreds of GPUs
  - Supports 1-bit SGD
    - significantly improves the performance during deep neural network (DNN) training using a single server with multiple GPUs and/or multiple servers with a single or multiple GPUs.
    - 1-bit SGD licensed separately from the rest of the lib and not licensed for commercial use.
  - ASGD is coming soon
- Supports most popular input data formats
  - Plain text (e.g. UCI datasets)
  - Speech formats (HTK)
  - Images
  - Binary
  - DSSM file format
  - New formats can be added by creating DataReader
- State of the art results on speech and image workloads

- First party Windows and Visual Studio support
  - Does NOT support OSX https://github.com/Microsoft/CNTK/issues/43
- First partysupport for Microsoft Azure
  - https://docs.microsoft.com/en-us/cognitive-toolkit/Deploy-Model-to-AKS
- CNTK allows the end users to fine-tune the system and trade off GPU memory for better computing efficiency
- For RNN of LSTM, CNTK obtains excellent time efficiency, which is up to 5-10 times better than other tools.
- Good model gallery/zoo
  - https://www.microsoft.com/en-us/cognitive-toolkit/features/model-gallery/



Torch/PyTorch
--------------
- PyTorch and Torch use the same C libraries that contain all the performance: TH, THC, THNN, THCUNN
- Facebook
- PyTorch
  - https://github.com/pytorch/pytorch
  - python
  - Watches 618, stars 10766, Fork 2235, Median Issue Resolution 2 days, Percent Open Issues 18%
  - Research Citations - 1
    - https://scholar.google.com/scholar?cites=7333966966476165372&as_sdt=5,33&sciodt=0,33&hl=en
  - PyTorch is great for research, experimentation and trying out exotic neural networks
  - Imperative Programing
    - Preforms computation as you typed it
    - There is no distinction between defining the computation graph and compiling
    - Most python code is Imperative
      - More flexible
      - You can treat your graphs more like typical python code and do things like use normal control structure (for and while loops) and debugging (adding print lines everywhere when you need it)
  - Dynamic Computation Graphing (Imperative Programming)
  - Above features make it best for research and development
  - Pretrained models
    - https://github.com/pytorch/vision
    - built in pretrained model loader in the python api
  - Visdom
    - Similar to TensorBoard
  - not designed with Production deployment in mind

- Torch
  - https://github.com/torch/torch7
  - Lua, C++
  - NYU
  - Watches 683, Stars 7577, Forks 2223, Median Issue Resolution 55 days, Percent Open Issues 33%
  - research Citations - 869
    - https://scholar.google.com/scholar?cites=5370776646193960982&as_sdt=40005&sciodt=0,10&hl=en
    - Torch7: A Matlab-like Environment for Machine Learning
    - Ronan Collobert, Koray Kavukcuoglu, Clement Farabet
    - 2011
  - Learning LUA is a big barrier for entry for most people
  - No autograd (writing your own backprop code)
  - more stable than PyTorch
  - More existing code and research projects to base your work off of.

- Most people would agree that you should be using PyTorch over Torch at this point.



caffe2
--------
- C++, Python
- Facebook
- Watches 488, Star 6684, Forks 1513, Median Issue Resolution 52 days, Percent Open Issues 49%
- https://developer.nvidia.com/caffe2

- Caffe2 is built to excel at mobile and at large scale deployments.
- multi-GPU
  - While it is new in Caffe2 to support multi-GPU, bringing Torch and Caffe2 together with the same level of GPU support, Caffe2 is built to excel at utilizing both multiple GPUs on a single-host and multiple hosts with GPUs.
- Caffe2 is headed towards supporting more industrial-strength applications with a heavy focus on mobile.

- Caffe2 improves Caffe 1.0 in a series of directions:
  - first-class support for large-scale distributed training
  - mobile deployment
  - new hardware support (in addition to CPU and CUDA)
  - flexibility for future directions such as quantized computation
  - stress tested by the vast scale of Facebook applications

- Mobile Support
 - Android and IOS



Keras
-----------
- Not a DL library per say, but a library that sits on top of other DL libs and provides a single, easy to use, high level interface
- Very modular, object oriented design
- Great for beginners, with great documentation
- Lacks in optimizations



Other
-----------
- Theano
  - Python
  - University of Montreal
  - Keras support as backed
  - May it rest in peace
    - https://groups.google.com/forum/#!msg/theano-users/7Poq8BZutbY/rNCIfvAEAwAJ
  - Research Citations - 139
    - Theano: A Python framework for fast computation of mathematical expressions
    - https://arxiv.org/find/cs/1/au:+Team_Theano_Development/0/1/0/all/0/1
    - 2016
  - Makes you do a lot of things from scratch, which leads to more verbose code.
  - Single GPU support
  - Numerous open-source deep-libraries have been built on top of Theano, including Keras, Lasagne and Blocks
    - https://github.com/fchollet/keras
    - https://lasagne.readthedocs.org/en/latest/
    - https://github.com/mila-udem/blocks
  - No real reason to use over TensorFlow unless you are working with old code.


- Paddle
  - https://github.com/PaddlePaddle/Paddle
  - Python
  - Baidu
  - Watches 545, Star 6191, Forks 1617, Median Issue Resolution 7 days, Percent Open Issues 24%
  - Chinese documentation with an English translation.

- Neon
  - https://github.com/PaddlePaddle/Paddle
  - python
  - Intel
  - Written with Intel MKL accelerated hardware in mind (Intel Xeon and Phi processors)
  - Watches 348, Stars 3366, Forks 754, Median Issue Resolution Time 28 days, Percent Open issues 16%

- Chainer
  - https://github.com/chainer/chainer
  - Python
  - Preferred Networks
    - https://www.preferred-networks.jp/ja/
  - Research Citations - 207
    - Chainer: a Next-Generation Open Source Framework for Deep Learning
    - http://learningsys.org/papers/LearningSys_2015_paper_33.pdf
    - 2015
  - Watches 296, Stars 3345, Forks 892, Median Issue Resolution Time 31 days, Percent Open issues 13%
  - Dynamic computation graph
  - Japanese and English Community

- Deeplearning4j
  - https://github.com/deeplearning4j/deeplearning4j
  - Java, Scala
  - Skymind
    - https://skymind.ai/
  - Keras Support (Python API)
  - Watches 777, Stars 8061, Forks 3937, Median Issue Resolution Time 19 days, Percent Open issues 21%
  - Written with Java and the JVM in mind
  - DL4J takes advantage of the latest distributed computing frameworks including Hadoop and Apache Spark to accelerate training. On multi-GPUs, it is equal to Caffe in performance.
  - Can import models from Tensoflow
  - Uses ND4J

- DyNet
  - C++, Python
  - https://github.com/clab/dynet
  - Carnegie Mellon University
  - Watches 177, Stars 2027, Forks 494, Median Issue Resolution Time 4 days, Percent Open issues 16%
  - Dynamic computation graph
  - Small user community

- Darknet
  - https://github.com/pjreddie/darknet
  - Python, C
  - Watches 481, Stars 5395, Forks 2592, Median Issue Resolution Time 55 days, Percent Open issues 78%
  - Very small open source effort with a laid back dev group
  - not useful for production environments

- Leaf
  - https://github.com/autumnai/leaf
  - Rust
  - autumnai
  - Watches 192, Stars 5158, Forks 262, Median Issue Resolution Time 131 days, Percent Open issues 58%
  - Support for the lib looks to be dead, many dead links in repo

- MatConvNet
  - https://github.com/vlfeat/matconvnet
  - Matlab
  - Watches 109, Stars 917, Forks 620, Median Issue Resolution Time 96 days, Percent Open issues 53%
  - a MATLAB toolbox implementing Convolutional Neural Networks (CNNs) for computer vision applications

- CoreML
  - https://developer.apple.com/machine-learning/
  - Swift, Objective-C
  - Apple
  - Closed source
  - Not a full DL library (you can not use it to train models at the moment), but mainly focused on deploying pretrained models to IOS and OSX devices
    - If you need to train your own model you will need to use one of the above libraries
    - Model converters available for Keras, Caffe, Scikit-learn, libSVM, XGBoost, MXNet, and TensorFlow