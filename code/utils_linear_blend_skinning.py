###############################################################################
### Perform linear blend skinning
### Adapted from https://github.com/vchoutas/smplx/blob/master/smplx/lbs.py
###
### Input : Shape, pose and all the learned parameters
### Output: 3D vertices, joints
###############################################################################

import torch
import torch.nn.functional as F


def lbs(beta, pose, V_, K_, S_, P_, J_, W_):
    # Get batch size
    bs = beta.shape[0]

    # Get device type ('cpu' or 'cuda')
    device = beta.device

    # Add shape contribution
    v_shaped = V_ + blend_shape(beta, S_) # [bs, 778, 3]

    # Get rest posed joints locations
    # NxJx3 array
    j_rest = vertices2joints(v_shaped, J_).contiguous() # [bs, 16, 3]

    # Add pose blend shapes
    # To convert 3 by 1 axis angle to 3 by 3 rotation matrix
    # Note: pose [bs, 16, 3] reshape to [bs*16, 3] for batch rodrigues
    eye3 = torch.eye(3, dtype=torch.float32, device=device) # Identity matrix
    rmats = batch_rodrigues(pose.view(-1, 3)).view(bs, -1, 3, 3) # [bs, 16, 3, 3]
    # pose_feature [bs, 135] where 135 = 15*9
    pose_feature = (rmats[:, 1:, :, :] - eye3).view([bs, -1])
    # pose_offsets [bs, 135] matmul [135, 778*3] = [bs, 778*3] -> [bs, 778, 3]
    pose_offsets = torch.matmul(pose_feature, P_).view(bs, -1, 3)
    v_posed      = v_shaped + pose_offsets # [bs, 778, 3]

    # Get global joint location
    # j_transformed [bs, 16, 3], A [bs, 16, 4, 4]
    j_transformed, A = batch_rigid_transform(rmats, j_rest, K_) 

    # Do skinning
    W = W_.unsqueeze(dim=0).expand([bs, -1, -1]) # [bs, 778, 16]
    # W[bs, 778, 16] matmul A[bs, 16, 16] = [bs, 778, 16] 
    # -> reshape to [bs, 778, 4, 4]
    T = torch.matmul(W, A.view(bs, -1, 16)).view(bs, -1, 4, 4) # [bs, 778, 4, 4]

    ones = torch.ones([bs, v_posed.shape[1], 1], 
        dtype=torch.float32, device=device) # [bs, 778, 1]
    v_posed_homo = torch.cat([v_posed, ones], dim=2) # [bs, 778, 4]
    # T[bs, 778, 4, 4] matmul v_posed_homo unsqueeze [bs, 778, 4, 1]
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1)) # [bs, 778, 4, 1]

    vertices = v_homo[:, :, :3, 0] # [bs, 778, 3]

    return vertices, j_transformed


def blend_shape(beta, S_):
    ''' 
    Calculates per vertex displacement due to shape deformation
    i.e. Multiply each shape displacement (shape blend shape) by 
         its corresponding beta and then sum them
    Displacement [b,v,t] = sum_{l} beta[b,l] * S_[v,t,l]

    Input:
        beta [b,l]   (batch size, length=10)
        S_   [v,t,l] (num of vert, 3, length=10)
    
    Output:
        blend_shape [b,v,t] (batchsize, num of vert, 3)
    '''
    return torch.einsum('bl,vtl->bvt', [beta, S_])


def vertices2joints(vert, J_):
    ''' 
    Calculates 3D joint positions from vertices
    using joint regressor array

    Input:
        vert [b,v,t] (batch size, num of vert, 3)
        J_   [j,v]   (num of joint, number of vert)

    Output:
        j_rest [b,j,t] (batch size, num of joint, 3)
    '''
    return torch.einsum('bvt,jv->bjt', [vert, J_])


def batch_rodrigues(rvecs, epsilon=1e-8):
    ''' 
    Calculates the rotation matrices for a batch of rotation vectors
    
    Input:
        rvecs [N,3] array of N axis-angle vectors
            
    Output:
        rmat [N,3,3] rotation matrices for the given axis-angle parameters
    '''
    # Get batch size
    bs = rvecs.shape[0]
    # Get device type
    device = rvecs.device

    angle = torch.norm(rvecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rvecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((bs, 3, 3), dtype=torch.float32, device=device)

    zeros = torch.zeros((bs, 1), dtype=torch.float32, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((bs, 3, 3))

    eye3 = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(dim=0)
    rmats = eye3 + sin * K + (1 - cos) * torch.bmm(K, K)
    
    return rmats


def transform_mat(R, t):
    ''' 
    Creates a batch of transformation matrices
    
    Input:
        R [B,3,3] array of a batch of rotation matrices
        t [B,3,1] array of a batch of translation vectors
    
    Output
        T [B,4,4] transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rmats, joints, parents):
    '''
    Applies a batch of rigid transformations to the joints
    
    Input:
        rmats   [B,N,3,3] rotation matrices
        joints  [B,N,3]   joint locations
        parents [B,N]     kinematic tree of each object

    Output:
        posed_joints   [B,N,3] 
            joint locations after applying the pose rotations

        rel_transforms [B,N,4,4] 
            relative (with respect to the root joint) 
            rigid transformations for all the joints
    '''
    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone() # [1, 16, 3, 1]
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(
        rmats.view(-1, 3, 3),
        rel_joints.view(-1, 3, 1)).view(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms