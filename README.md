miml：MeteoInfo machine learning toolbox
========================================

Installation
------------

Copy miml folder into MeteoInfo -> toolbox folder.

Features
--------

Machine learning algorithms of classification, regression, clustering based on [Smile](https://haifengl.github.io/smile/),
and neural network model based on [Encog](https://www.heatonresearch.com/encog/).

Example
-------

K-Means clustering:

    from miml import datasets
    from miml.cluster import KMeans

    fn = os.path.join(datasets.get_data_home(), 'clustering', 'gaussian', 
        'six.txt')
    df = DataFrame.read_table(fn, header=None, names=['x1','x2'], 
        format='%2f')
    x = df.values

    model = KMeans(6, runs=20)
    y = model.fit_predict(x)

    scatter(x[:,0], x[:,1], c=y, edgecolor=None, s=3)
    title('K-Means clustering example')
    
![K-Means](http://www.meteothink.org/_images/kmeans_1.png)

Documentation
-------------

Learn more about MeteoInfo in its official documentation at http://meteothink.org/

License
-------

Copyright 2019, miml Developers

Licensed under the LGPL License, Version 3.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.gnu.org/licenses/lgpl.html

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.