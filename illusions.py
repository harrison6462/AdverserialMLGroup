import cv2
import numpy as np
import os
def is_safe_access(img, x : int, y : int):
  w, h = img.shape[1], img.shape[0]
  return not (x >= w or x < 0 or y < 0 or y >= h)

def get_pixel_safe(img, x, y):
  w, h = img.shape[1], img.shape[0]
  if not is_safe_access(img, x, y):
    return 0
  return img[y][x]

def set_pixel_safe(img, x : int, y : int, val : int):
  w, h = img.shape[1], img.shape[0]
  if not is_safe_access(img, x, y):
    print('Failed to set', x, y, 'pixel of img size', (w,h))
    return
  img[y][x] = val

# requires y0 < maxy
# returns array of all significant points between y0, maxy of triangle wave
def triangle_wave_pts(y0, maxy, amp, pd, xoff):
  ps = [] #list of points (x,y)
  xs = [xoff, xoff+amp, xoff, xoff-amp] #array of extrema
  stg = 0
  y = y0
  while y < maxy + pd:
    ps = np.append(ps, [xs[stg], y])
    stg = (stg+1) % 4
    y += pd/4
  return ps.reshape((-1, 1, 2))

def test_draw_cat():
  def make_zigzags(w, h, bwidth=10, wwidth=15, pd=30, x0=0, y0=0):
    img = np.zeros([h, w], dtype=np.uint8)
    img.fill(255)
    x = x0
    while x < w:
      #draw polylines w/o connecting first/last points w/ width of 3 and color black
      cv2.polylines(img, np.int32([triangle_wave_pts(y0, h, bwidth, pd, x)]), False, (0,0,0), 3)
      x += bwidth + wwidth
    return img
  cat = cv2.cvtColor(cv2.imread('cat.png'), cv2.COLOR_BGR2GRAY)
  i = make_zigzags(cat.shape[1], cat.shape[0])
  cv2.imshow('cat', cv2.addWeighted(i, .3, cat, .7, 0))  
  cv2.waitKey(0)
  cv2.destroyAllWindows()

#img, heatmap are cv2 images, p1, p2 are bounding box
def draw_heatmapped_thickness_line(img, heatmap, p1, p2, thickness, offsetfn, color = (0,0,0)):
  px0, px1, py0, py1 = p1[0], p2[0], p1[1], p2[1]
  # determine bounding box over interval
  x0, y0, x1, y1 = int(min(px0, px1)), int(min(py0, py1)), int(max(px0,px1)), int(max(py0,py1))

  #average pixel value on interval
  accum = 0.0
  for x in range(x0, x1):
    for y in range(y0, y1):
      accum += get_pixel_safe(heatmap, x, y)
  accum /= ((x1-x0+1)*(y1-y0+1))
  
  off = int(min(1, offsetfn (accum)))
  return cv2.line(img, (int(p1[0]-off), int(p1[1])), (int(p2[0]-off), int(p2[1])), color, thickness)

def heatmapped_zigzags(heatmap, bwidth=4, wwidth=3, thickness=2, pd=15, x0=0, y0=0):
  w, h = heatmap.shape[1], heatmap.shape[0]
  img = np.zeros([h, w])
  img.fill(255)
  x = x0
  while x < w+bwidth:
    pts = triangle_wave_pts(y0, h, bwidth, pd, x)
    for i in range(0, len(pts)-1):
      img = draw_heatmapped_thickness_line(img, heatmap, pts[i][0], pts[i+1][0], thickness, lambda accum : (accum/50))
    x += bwidth + wwidth
  return img

def pil_to_cv2(mnist):
  return np.array(mnist)

def mnist_test():
  import torchvision.datasets as datasets
  mnist_trainset = datasets.MNIST(root='./data', train=True, download=True,transform=None)
  for i in range(len(mnist_trainset)):
    cv2.imwrite(os.path.join('C:/Users/harrison/Desktop/illusions/illusionmnist', str(i)+'.jpg'), heatmapped_zigzags(pil_to_cv2(mnist_trainset[i][0].resize((128,128)))))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
#mnist_test()

def cifar_test():
  import torchvision.datasets as datasets
  cifar_trainset = datasets.OxfordIIITPet(root='./pet', download=True)
  #for i in range(1):
  img = pil_to_cv2(cifar_trainset[0][0])
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  sobel = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)
  sobel = np.vectorize(lambda x : abs(x))(sobel).astype(np.uint8)
  #cv2.imwrite(os.path.join('C:/Users/harrison/Desktop/illusions/cifar', str(i)+'.jpg'), heatmapped_zigzags(pil_to_cv2(sobelx)))
  cv2.imshow('orig', img)
  cv2.imshow('b&w', sobel)
  cv2.imshow('cifar', heatmapped_zigzags(pil_to_cv2(sobel)))
  cv2.waitKey(0)
  cv2.destroyAllWindows()
cifar_test()