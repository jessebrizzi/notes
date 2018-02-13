API Services
----------------
- ⭐ logograb
    - Prob best option for paying for the service
    - https://www.logograb.com/products/brand-detection-api
    - image and video API
    - 10k brand catalog
    - "Add any brand logo in minutes"
- google vision API
    - https://cloud.google.com/vision/docs/detecting-logos
    - only predefined list of "popular brands"
    - can't find the list of brands
- Amazon Rekognition
    - does not support searching for logos
    - https://forums.aws.amazon.com/thread.jspa?threadID=250153
    - would have to train a object detector model
- Clarifai
    - https://clarifai.com/models/logo-image-recognition-model/c443119bf2ed4da98487520d01a0b1e3
    - 500 brands
    - Considered in Beta
- Brandwatch
    - Has the technology for logo detection, not sure if they expose it directly as an API
    - Part of their brand image insights product
- GumGum
    - Same situation as Brandwatch, has the tech for logo detection, but it's in their brand insights product and not exposed as an api


Logo Datasets
---------------
| dataset              | brands/logos | logo Images | RoIs     | Website                                                                         |
| -------------------- | ------------ | ----------- | -------- | ------------------------------------------------------------------------------- |
| BelgaLogos           | 37           | 1,321       | 2,697    | http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/BelgaLogos.html          |
| FlickrBelgaLogos     | 37           | 2,697       | 2,697    | http://www-sop.inria.fr/members/Alexis.Joly/BelgaLogos/FlickrBelgaLogos.html    |
| FlickrLogos-27       | 27           | 810         | 1,261    | http://image.ntua.gr/iva/datasets/flickr_logos/                                 |
| FlickrLogos-32       | 32           | 2,240       | 3,404    | http://www.multimedia-computing.de/flickrlogos/                                 |
| Logos-32plus         | 32           | 7,830       | 12,300   | http://www.ivl.disco.unimib.it/activities/logo-recognition/                     |
| TopLogo10            | 10           | 700         | 863      | http://www.eecs.qmul.ac.uk/~hs308/qmul_toplogo10.html/                          |
| Logos-160            | 160          | 73,414      | 130,608  | http://logo-net.org/                                                            |
| Logos-18             | 18           | 8,460       | 16,043   | http://logo-net.org/                                                            |
| Logos in the Wild    | 871          | 11,054      | 32,850   | https://www.iosb.fraunhofer.de/servlet/is/78045/                                |


Open CV Approaches
-------------------
- matchTemplate
    - https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    - Bad at dealing with scale and rotation
    - Fastest, but not a real option
- Classic OpenCV Feature extraction, matching, and Homography for Detection and Localization
    - http://cs.uccs.edu/~gsc/pub/master/yalsahaf/doc/YousefAlsahafiReport.pdf
    - local invariant descriptors
        - (SIFT, SURF, ORB, FAST)
        - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_table_of_contents_feature2d/py_table_of_contents_feature2d.html
        - SIFT and SURF are not free for commercial use, require license.
        - Issue with these features is that they are all corner based and when applied to simple logos like the nike swoosh can not find enough features to match against the query image.
    - Feature Matching
        - (Brute-Force or FLANN methods)
        - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    - Homography
        - (RANSAC or LMEDs)
        - http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html
- HAAR Cascade
    - Requires training of the model with positive and negative samples
    - Closed Set approach
    - Might as well go deep learning for the improved accuracy and buzz words


Papers
----------------
- On the Benefit of Synthetic Data for Company Logo detection
    - Multimedia Computing and Computer Vision Lab University of Augsburg
    - http://www.multimedia-computing.de/mediawiki/images/c/cf/ACMMM2015.pdf
    - 2015
    - Used CNN to extract features from logos and determined their brand by classification with a observation_timestamp
    - Uses Synthetic data generation (Color, perspective, background replacement)
- DeepLogo: Hitting Logo Recognition with the Deep Neural Network Hammer
    - ASPIRE Laboratory, University of California, Berkeley
    - https://arxiv.org/pdf/1510.02131.pdf
    - 7 Oct 2015
    - Uses AlexNet/VGG Fast R-CNN
    - Trains and tests on FlickrLogos-32
    - Closed Set Approach (only works on the logos in the training set)
- Automatic Graphic Logo Detection via Fast Region-based Convolutional networks
    - Center for Informatics and Systems of the University of Coimbra
    - https://arxiv.org/pdf/1604.06083.pdf
    - 20 April 2016
    - Uses VGG F-RCN with selective search
    - Trains and tests on FlickrLogos-32
    - Closed Set Approach
- ⭐ Open Set Logo Detection and Retrieval
    - Fraunhofer IOSB, Karlsruhe, Germany
    - https://arxiv.org/pdf/1710.10891.pdf
    - 30 Oct 2017
    - Claims to be the first approach at looking at the logo detection/recognition problem as an open set problem as opposed to a closed set (IE training on a dataset of N known brand logos, and only ever detecting those same logos).
    - Introduces the Logos in the Wild Dataset
    - Proposed method uses 2 networks, a general logo detection network, and then a logo feature encoding network to map the query logo to the detected logos.
- ⭐ Scalable Object Detection for Stylized Objects
    - Microsoft AI & Research Munich
    - https://arxiv.org/pdf/1711.09822.pdf
    - 29 Nov 2017
    - The only paper that sights the Open Set Logo paper.
    - Proposed method is the same as above, but they include triplet training for the feature encoding network (so essentially what we do for product search)
    - Introduces its own dataset, MSR1k, but it does not look like it is publicly available.