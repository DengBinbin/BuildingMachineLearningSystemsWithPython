import numpy as np
import mahotas as mh
image = mh.imread('scene00.jpg')
from matplotlib import pyplot as plt
import pylab
import sys


# plt.imshow(image)
# plt.show()


#matplotlib默认将单通道图像转换成了假彩色图像，较高值用红色，较低值用蓝色
image = mh.colors.rgb2grey(image, dtype=np.uint8)
plt.imshow(image) # Display the image

#现在图像变成灰度图了
plt.gray()
#将图像输入给otsu方法，该方法会找到合适的域值
thresh = mh.thresholding.otsu(image)
print('Otsu threshold is {}.'.format(thresh))
# Otsu threshold is 138.

plt.imshow(image > thresh)


#高斯滤波器，其中16为滤波器的大小（滤波器的标准差），越大越模糊
im16 = mh.gaussian_filter(image,1)

#换一张新的图片
im = mh.demos.load('lenna')

#加入椒盐噪声
salt = np.random.random(im.shape) > .975
pepper = np.random.random(im.shape) > .975
im_salt = mh.stretch(im)
im_salt = np.maximum(salt*170, im_salt)
im_salt = np.minimum(pepper*30 + im_salt*(~pepper), im_salt)
plt.imshow(im_salt)
# plt.show()
#聚焦中心
r,g,b = im.transpose(2,0,1) 
r12 = mh.gaussian_filter(r, 12.)
g12 = mh.gaussian_filter(g, 12.)
b12 = mh.gaussian_filter(b, 12.)
im12 = mh.as_rgb(r12,g12,b12)

h, w = r.shape # height and width
Y, X = np.mgrid[:h,:w]
Y = Y-h/2. # center at h/2
Y = Y / Y.max() # normalize to -1 .. +1

X = X-w/2.
X = X / X.max()

C = np.exp(-2.*(X**2+ Y**2))

# Normalize again to 0..1
C = C - C.min()
C = C / C.ptp()
C = C[:,:,None] # This adds a dummy third dimension to C

ringed = mh.stretch(im*C + (1-C)*im12)
plt.imshow(ringed)
# plt.show()

#计算图像特征,haralick_features是一个4*13数组，第一维代表4个可能的方向（上下左右）
haralick_features = mh.features.haralick(image)
#如果我们对方向不感兴趣，那么可以对方向（即第一维进行算术平均）
haralick_features_mean = np.mean(haralick_features, axis=0)
#将原始向量(4,13)进行拼接，变成一个(52,)的向量
haralick_features_all = np.ravel(haralick_features)
print(haralick_features_all.shape) #(52,)

#有了特征我们就可以去对图像进行分类
from glob import glob
images = glob('../SimpleImageDataset/*.jpg')
features = []
labels = []
for im in images:
  labels.append(im[:-len('00.jpg')])
  im = mh.imread(im)
  im = mh.colors.rgb2gray(im, dtype=np.uint8)
  features.append(mh.features.haralick(im).ravel())

features = np.array(features)  #(90,52)
labels = np.array(labels)      #(90,)
#训练一个Logistic 回归模型
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
clf = Pipeline([('preproc', StandardScaler()),
                ('classifier', LogisticRegression())])

from sklearn import cross_validation
cv = cross_validation.LeaveOneOut(len(images))
scores = cross_validation.cross_val_score(
    clf, features, labels, cv=cv)
print('Accuracy: {:.1%}'.format(scores.mean()))
# Accuracy: 81.1%

#增加特征
def chist(im):
    im = im // 64
    r,g,b = im.transpose((2,0,1))
    pixels = 1 * r + 4 * b + 16 * g
    hist = np.bincount(pixels.ravel(), minlength=64)
    hist = hist.astype(float)
    hist = np.log1p(hist)
    return hist

features = []
for im in images:
  im = mh.imread(im)
  features.append(chist(im))

features = []
for im in images:
  imcolor = mh.imread(im)
  im = mh.colors.rgb2gray(imcolor, dtype=np.uint8)
  features.append(np.concatenate([
          mh.features.haralick(im).ravel(),
          chist(imcolor),
      ]))

scores = cross_validation.cross_val_score(
    clf, features, labels, cv=cv)
print('Accuracy: {:.1%}'.format(scores.mean()))
# Accuracy: 95.6%

#忽略边缘的信息
features = []
for im in images:
  imcolor = mh.imread(im)
  # Ignore everything in the 200 pixels close to the borders
  imcolor = imcolor[200:-200, 200:-200]
  im = mh.colors.rgb2gray(imcolor, dtype=np.uint8)
  features.append(np.concatenate([
          mh.features.haralick(im).ravel(),
          chist(imcolor),
      ]))
#对特征做归一化
sc = StandardScaler()
features = sc.fit_transform(features)
from scipy.spatial import distance
dists = distance.squareform(distance.pdist(features))


fig, axes = plt.subplots(2, 9)
for ci,i in enumerate(range(0,90,10)):
    left = images[i]
    dists_left = dists[i]
    right = dists_left.argsort()
    # right[0] is the same as left[i], so pick the next closest element
    right = right[1]
    right = images[right]
    left = mh.imread(left)
    right = mh.imread(right)
    axes[0, ci].imshow(left)
    axes[1, ci].imshow(right)



from sklearn.grid_search import GridSearchCV
C_range = 10.0 ** np.arange(-4, 3)
grid = GridSearchCV(LogisticRegression(), param_grid={'C' : C_range})
clf = Pipeline([('preproc', StandardScaler()),
               ('classifier', grid)])

cv = cross_validation.KFold(len(features), 5,
                     shuffle=True, random_state=123)
scores = cross_validation.cross_val_score(
   clf, features, labels, cv=cv)
print('Accuracy: {:.1%}'.format(scores.mean()))



from mahotas.features import surf
image = mh.demos.load('lena')
image = mh.colors.rgb2gray(image, dtype=np.uint8)
descriptors = surf.surf(image, descriptor_only=True)

from mahotas.features import surf
descriptors = surf.dense(image, spacing=16)
alldescriptors = []
for im in images:
  im = mh.imread(im, as_grey=True)
  im = im.astype(np.uint8)
  alldescriptors.append(surf.dense(image, spacing=16))
# get all descriptors into a single array
concatenated = np.concatenate(alldescriptors)
print(concatenated.shape)
print('Number of descriptors: {}'.format(
       len(concatenated)))
# use only every 64th vector
concatenated = concatenated[::64] 
print(concatenated.shape)

from sklearn.cluster import KMeans # FIXME CAPITALIZATION
k = 256
km = KMeans(k)
km.fit(concatenated)

features = []
for d in alldescriptors:
  c = km.predict(d)
  features.append(
      np.array([np.sum(c == ci) for ci in range(k)])
  )
# build single array and convert to float
features = np.array(features, dtype=float)

scores = cross_validation.cross_val_score(
   clf, features, labels, cv=cv)
print('Accuracy: {:.1%}'.format(scores.mean()))
# Accuracy: 62.6%


