# Code based on an outline in the paper:
# http://www.ipol.im/pub/art/2014/82/

from skimage.feature import peak_local_max
import cv2
import math
import numpy as np
import scipy
π = math.pi
from math import sin, cos
# utility functions



##### for debudgging
def test1(list_of_np_arrays):
    for i in list_of_np_arrays:
        print(i.shape)
def print_all(list1):
    for i in list1:
        print(i)

##### for utilities
#distance of 2 np arrays
def distance(x1, x2): return np.sum(np.sqrt((x1 - x2) * (x1 - x2)))
# allows for triple elmentwise comparison
def np_3and(x, y, z): np.logical_and(x, np.logical_and(y, z))
# gets a 3x3 slice of a 3d matrix at the given center.
def cube_slice(array,x,y,z): return array[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]

##### Parameters
σ_in = 0.5
σ_min = 0.8
delta_min=.5
n_spo = 3
n_oct = 5
λori = 1.5
λ_descr = 6
resolution = 0.05
nbins= 36
C_DoG = 0.015  # CDoG = 0.015 * (2**(1 / n_spo) - 1) / (2**(1 / 3) - 1) if n_spo != 3
C_edge=10
# The scale for each octive, o
def scale(o): return delta_min*2**(o)

##############################################################

def Compute_the_Gaussian_scale_space(img):
    img=img.astype(float)
    # u′ ← bilinear interpolation(uin, δmin)
    v00 = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2), interpolation=cv2.INTER_CUBIC)
    v00 = cv2.GaussianBlur(v00, (5, 5), (σ_min**2-σ_in**2)**.5/delta_min)
    v_layer = [v00]
    for s in range(1, n_spo + 3):
        σ  = σ_min/delta_min*(2**(2*(s+1)/n_spo)-2**(2*s/n_spo))**.5
        bounds=int((σ*8//2)*2+3)
        v_layer += [cv2.GaussianBlur(v_layer[-1], (bounds, bounds), σ,σ) ]
    v_scale_space = [np.stack(v_layer).astype(float)]
    for o in range(1, n_oct+1):
        shapes=v_scale_space[-1][n_spo].shape
        v0 = cv2.resize(v_scale_space[-1][n_spo], (shapes[1] // 2, shapes[0] // 2),
                        interpolation=cv2.INTER_NEAREST)
        v_layer = [v0]
        for s in range(1, n_spo + 3):
            σ  = σ_min/delta_min*(2**(2*(s+1)/n_spo)-2**(2*s/n_spo))**.5
            bounds=int((σ*8//2)*2+3)
            v_layer += [cv2.GaussianBlur(v_layer[-1], (bounds, bounds), σ,σ) ]
        v_scale_space += [np.stack(v_layer).astype(float)]
    return v_scale_space

##############################################################

def Compute_the_Difference_of_Gaussians(v_scale_space):
    return [v_scale_space[o][1:n_spo + 3] - v_scale_space[o][0:n_spo + 2]
            for o in range(0, len(v_scale_space)-1)]

##############################################################

def Find_2d_maxs(i,DoG):
    #for use in Find_discrete_extrema
    maxs_2d=[]
    for s in range(1, len(DoG) - 1):
        local_maxs = peak_local_max(DoG[s], min_distance=2, indices=True)
        for imax in range(len(local_maxs)):
            x = local_maxs[imax][0]
            y = local_maxs[imax][1]
            if abs(DoG[s, x, y])>=C_DoG*.8: # remove low-contrast points
                if np.amax(cube_slice(DoG,s,x,y)) == DoG[s, x, y]:
                    maxs_2d += [(s, x, y)]
    return maxs_2d

def Find_discrete_extrema(w_DoG):
    CDoG = 0.015
    CDoG = [CDoG * (2**(1 / i) - 1) / (2**(1 / 3) - 1)
            for i in range(1, n_oct + 1)]
    extrema = []
    for i, DoG in enumerate(w_DoG):
        extrema+=[(i,Find_2d_maxs(i, DoG)+Find_2d_maxs(i,-DoG))]
    print("number of extrema=", str(sum([len(x[1]) for x in extrema])))
    return extrema


##############################################################

def quadratic_interp(w,o,s,m,n):
    #### for use in Refine_candidate_keypoints_location_with_sub_pixel_precision
    ###
    # note that for α in αs where :
    #     αs = [(i,j,k) for i in [-1/2,1/2] for j in [-1/2,1/2] for k in [-1/2,1/2]]
    # the quadratic fit is given by:
    #     ω(α)= w + np.matmul(α.T, g)+1/2*np.matmul(α.T, np.matmul(H, α))

    # Define G
    g=np.array([(w[s+1,m,n]-w[s-1,m,n])/2,
                (w[s,m+1,n]-w[s,m-1,n])/2,
                (w[s,m,n+1]-w[s,m,n-1])/2])

    # define H, the Hessian matrix
    mid=w[s,m,n]
    h11=w[s+1,m,n]+w[s-1,m,n]-2*mid
    h22=w[s,m+1,n]+w[s,m-1,n]-2*mid
    h33=w[s,m,n+1]+w[s,m,n-1]-2*mid
    h12=(w[s+1,m+1,n]+w[s-1,m-1,n]-w[s+1,m-1,n]-w[s-1,m+1,n])/4
    h13=(w[s+1,m,n+1]+w[s-1,m,n-1]-w[s+1,m,n-1]-w[s-1,m,n+1])/4
    h23=(w[s,m+1,n+1]+w[s,m-1,n-1]-w[s,m+1,n-1]-w[s,m-1,n+1])/4
    H=np.array([[h11,h12,h13],
                [h12,h22,h23],
                [h13,h23,h33]])

    # compute the location of the zero
    alpha_zero = -np.matmul(np.linalg.inv(H),g)

    ### Now, do some filtering while we have all this info computed
    # discard if alpha_zero is further than .5 from the origin
    if len(np.where(abs(alpha_zero)>0.6)[0]) != 0: return False

    #calculate ω and filter noise
    ω = mid + 1/2*np.matmul(g, alpha_zero)
    if abs(ω) < C_DoG: return False

    #filter out edges
    edgeness=(h22+h33)**2/(h22*h33-h23*h23)
    if edgeness > (C_edge+1)**2/C_edge: return False

    ### return kp details
    σ=scale(o)/delta_min*σ_min*2**((alpha_zero[0]+s)/n_spo)
    x, y = scale(o)*(alpha_zero[1]+m),scale(o)*(alpha_zero[2]+n)


def Refine_candidate_keypoints_location_with_sub_pixel_precision(w_DoG, extrema_of_DoG):
# most of the machinery is in the quadratic_interp funtion
    kps = []
    for o, extrema in extrema_of_DoG:
        w=w_DoG[o]
        for s, m, n in extrema:
            kp = quadratic_interp(w,o,s,m,n)
            if kp is not False: kps += [kp]
#     print("number of keypoints = ", len(kps))
    return kps

##############################################################
# not fully implemented.  too much seems reuable from keypoint extraction too use seperate functions,
# but the implementation has been sloppy
def Build_the_keypoints_descriptor(test_v_scale_space, test_keypoints):
    new_kps = []
    descriptors = []
    for o, s, i, j, σ, x, y, ω in test_keypoints:
        thresh = 3 * λori * σ
        bound = thresh * 2 ** 0.5

        #   find_bounds
        m_upper = int((x + bound) / scale(o)) + 1
        m_lower = int((x - bound) / scale(o))
        n_upper = int((y + bound) / scale(o)) + 1
        n_lower = int((y - bound) / scale(o))

        if m_lower - 1 < 0 or n_lower - 1 < 0 : continue
        v = test_v_scale_space[o][s]
        if m_upper+1 > v.shape[0] or n_upper+1 > v.shape[1] : continue

        local_grid = v[m_lower:m_upper,n_lower:n_upper]
    #         local_dist = np.zeros_like(local_grid)
    #         local_m_range,local_n_range = np.where(local_grid)

        try:
            local_m_grad_grid = v[m_lower+1:m_upper+1,n_lower:n_upper]-v[m_lower-1:m_upper-1,n_lower:n_upper]
            local_n_grad_grid = v[m_lower:m_upper,n_lower+1:n_upper+1]-v[m_lower:m_upper,n_lower-1:n_upper-1]
        except:
            print(m_upper,m_lower,n_upper,n_lower)
            print(v[m_lower+1:m_upper+1,n_lower:n_upper])
            print(v[m_lower-1:m_upper-1,n_lower:n_upper])
            print(v[m_lower:m_upper,n_lower+1:n_upper+1])
            print(v[m_lower:m_upper,n_lower-1:n_upper-1])
        local_grad_angle  = np.arctan2(local_m_grad_grid, local_n_grad_grid) % 2 * π
        local_grad_abs    = np.sqrt(local_m_grad_grid**2 + local_n_grad_grid**2)

        # Create index arrays
        m_upper_thresh = int((x + thresh) / scale(o)) + 1
        m_lower_thresh = int((x - thresh) / scale(o))
        n_upper_thresh = int((y + thresh) / scale(o)) + 1
        n_lower_thresh = int((y - thresh) / scale(o))

        m_range,n_range=np.where()
        m_range += (m_lower_thresh-m_lower)
        n_range += (n_lower_thresh-n_lower)
        dist_sq = ((m_range+m_lower)*scale(o)-x)**2+((n_range+n_lower)*scale(o)-y)**2
        dist = np.sqrt(dist_sq)

        # Calculate info for contribution
        grad_angle = local_grad_angle[m_range,n_range]
        grad_abs = local_grad_abs[m_range,n_range]
        contrib = np.exp(dist_sq / (-2*(λori * σ)**2))*grad_abs

        # Create histogram
        hist, bins = np.histogram(grad_angle, bins=nbins, range=(0, 2 * π), normed=None, weights=contrib)
        extended_hist = np.tile(hist, 5)
        kernel = np.array([1 / 3, 1 / 3, 1 / 3])
        for _ in range(6):
            extended_hist = np.convolve(extended_hist, kernel, mode='same')
        hist_new = extended_hist[2 * len(hist)-1:3 * len(hist) + 1]
        hist_max = np.amax(hist_new)
        hist_len=len(hist)
        # find reference angle(s)
        ref_oris = np.where(hist_new >= 0.8 * hist_max)[0]
        ref_oris = [ori for ori in ref_oris if not(ori == 0 or ori == hist_len+1)]
        ref_oris = [(ori,hist_new[ori - 1],hist_new[ori],hist_new[ori + 1])
                    for ori in ref_oris
                    if not (hist_new[ori] < hist_new[ori + 1]
                         or hist_new[ori] < hist_new[ori - 1])]
        theta_refs=[]
        for ori,low,mid,high in ref_oris:
            theta_refs += [(bins[ori]+π/nbins*(low-high)/(high+low-2*mid)) % 2*π]
            new_kps   += [(o, s, i, j, σ, x, y, ω, theta_refs)]

        ####################################
        #and roll right into creating a descriptor

        # Create index arrays
        m_index, n_index = np.where(local_grid)
        print(local_grid.shape[0]*local_grid.shape[1],len(m_index),len(n_index))
        xkey = x - m_lower * scale(o)
        ykey = y - n_lower * scale(o)
        m_hat = m_index*scale(o)-xkey
        n_hat = n_index*scale(o)-ykey
        for θkey in theta_refs:
            x_hat =  m_hat * (cos(θkey)/σ) + n_hat * (sin(θkey)/σ)
            y_hat = -m_hat * (sin(θkey)/σ) + n_hat * (cos(θkey)/σ)
    #             print(np.where(np.logical_and(abs(x_hat)<=λ_descr, abs(y_hat)<=λ_descr)))
            index_hat = np.where(np.logical_and(abs(x_hat)<=λ_descr, abs(y_hat)<=λ_descr))
            m_index_hat=m_index[index_hat]
            n_index_hat=n_index[index_hat]
            assert len(n_index_hat) != 0

            # Calculate distance info for contribution
            dist_sq_hat = (m_hat) ** 2 + (n_hat) ** 2

            # Calculate info for contribution
            theta_grad_hat = (local_grad_angle[(m_index_hat,n_index_hat)] - θkey) % (2 * π)


            grad_abs_hat = local_grad_abs[m_index,n_index]
            contrib_hat = np.exp(dist_sq_hat / (-2*(λ_descr * σ)**2))*grad_abs_hat

            n_hist = 4
            n_ori  = 8
            feature_vector = np.zeros((n_hist, n_hist, n_ori)).astype(float)
            factor = 2 * λ_descr / n_hist
            for ii in range(n_hist):
                for ij in range(n_hist):
                    x_hist = (2 * ii - (1 + n_hist)) * λ_descr / n_hist
                    y_hist = (2 * ij - (1 + n_hist)) * λ_descr / n_hist
                    for k in range(0, n_ori):
                        theta_center = 2 * π * (k-1) / n_ori
                        points_close = np.where(np_3and(
                            abs(m_index_hat - x_hist) < factor,
                            abs(n_index_hat - y_hist) < factor,
                            abs(theta_grad_hat - theta_center) % 2 * π < 2 * λ_descr / n_hist))
                        contribution = ((1-(m_index_hat[points_close]-x_hist) / factor) *
                                        (1-(n_index_hat[points_close]-y_hist)/factor) *
                                        (1-(theta_grad_hat[points_close]-theta_center)*n_ori/(2*λ_descr)) *
                                        contrib_hat[points_close])
                        feature_vector[ii, ij, k] = np.sum(contribution)
                norm = .2 * (np.sum(feature_vector * feature_vector)**.5)
                feature_vector[feature_vector >= norm] = norm
                descriptors += [feature_vector.flatten()]
    return descriptors

##############################################################
## not fully implemented
def Find_Match(descriptors1, descriptors2):
    matches = []
    for d1 in descriptors1:
        distances = []
        for d2 in descriptors2:
            distances += [distance(d1, d2)]
        nearest = sorted(distances, reverse=True)[0:5]
        if nearest[0] >= nearest[2] * .8:
            matches += [[d1, d2]]
    return matches

## alt Find_Match function
# def Find_Match(descriptors1,descriptors2):
#     distances=np.array([[distance(d1, d2) for d2 in descriptors2] for d1 in descriptors1])
#     return distances

#########################################

if __name__ == '__main__':
    input_filename="test_pic.jpg"
    output_filename="sift_test1.jpg"
    ####### begining
    # read image **AS A FLOAT**
    img = cv2.imread(input_filename, cv2.IMREAD_GRAYSCALE).astype(float)
    # compute Guassian scale space
    v_scale_space = Compute_the_Gaussian_scale_space(img)
    # use scale space to approximate the DoG
    w_DoG = Compute_the_Difference_of_Gaussians(v_scale_space)
    # use the DoG to compute extreema
    extrema = Find_discrete_extrema(w_DoG)
    # compute sub-pixel location of extreema.
    keypoints = Refine_candidate_keypoints_location_with_sub_pixel_precision(w_DoG, extrema)
    #### old_function, combined with descriptor constructor
    ### keypoints = Assign_a_reference_orientation_to_each_keypoint(v_scale_space, keypoints)
    # compute keypoints and descriptors simultaiously.
    descriptors = Compute_the_keypoints_and_descriptor(v_scale_space, keypoints)
    # finally compute the euclidian distance between discriptors (not funnly implemented)
    matches = Find_Match(descriptors1,descriptors2)
    # finally compute the euclidian distance between discriptors
    cv2.imwrite(output_filename,matches)
    ######## end
    print("done")
