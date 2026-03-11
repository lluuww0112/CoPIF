# download mscoco base images
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip -d ./images

# download annotations : refcoco
wget https://web.archive.org/web/20220413011718/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
# download annotations : refcoco+
wget https://web.archive.org/web/20220413011656/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip
# download annotations : refcocog
wget https://web.archive.org/web/20220413012904/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip


unzip refcoco.zip
unzip refcoco+.zip
unzip refcocog.zip