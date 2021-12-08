###############################################################################
### Make use of Max Planck Institue MANO model
### Ensure the MANO_RIGHT.pkl file is in the /model folder
###
### Input : Shape and pose parameters
### Output: 3D mesh vertices and joints
###############################################################################

import cv2
import torch
import pickle
import numpy as np
import torch.nn as nn

from utils_linear_blend_skinning import lbs


class MANO(nn.Module):
    def __init__(self):
        super(MANO, self).__init__()

        #################################
        ### Load parameters from file ###
        #################################
        file_path = '../model/MANO_RIGHT.pkl'
        dd = pickle.load(open(file_path, 'rb'), encoding='latin1')

        # Standardize naming convention to a single capital letter
        self.V = dd['v_template']       # Vertices of template model (V)
        self.F = dd['f']                # Faces of the model (F)
        K      = dd['kintree_table'][0] # Kinematic tree defining the parent joint (K)
        S      = dd['shapedirs']        # Shape blend shapes that are learned (S)
        P      = dd['posedirs']         # Pose blend shapes that are learned (P)
        J      = dd['J_regressor']      # Joint regressor that are learned (J)
        W      = dd['weights']          # Weights that are learned (W)
        C      = dd['hands_components'] # Components of hand PCA (C)
        M      = dd['hands_mean']       # Mean hand PCA pose (M)

        # Original parameter size and data type
        # V (778, 3)      float64
        # F (1538, 3)     uint32
        # K (16,)         int64
        # S (778, 3, 10)  float64
        # P (778, 3, 135) float64
        # J (16, 778)     float64
        # W (778, 16)     float64
        # C (45, 45)      float64
        # M (45,)         float64


        ########################
        ### Convert to numpy ###
        ########################
        self.V = np.array(self.V, dtype=np.float32)
        self.F = np.array(self.F, dtype=np.int32) # Need to convert from uint32 to int32 to allow interation in v0 = vertices[:, self.F[:,0],:] # [bs, 1538, 3]
        S      = np.array(S, dtype=np.float32)
        P      = np.array(P, dtype=np.float32)
        J      = np.array(J.todense(), dtype=np.float32) # Need to convert sparse to dense matrix
        W      = np.array(W, dtype=np.float32)
        C      = np.array(C, dtype=np.float32) # Use PCA
        # C      = np.eye(45, dtype=np.float32) # Directly modify 15 x 3 articulation angles
        M      = np.array(M, dtype=np.float32)
        # Convert undefined parent of root joint to -1
        K[0] = -1
        # Reshape the pose blend shapes from (778, 3, 135) -> (778*3, 135) -> (135, 778*3)
        num_pose_basis = P.shape[-1]
        P = np.reshape(P, [-1, num_pose_basis]).T


        ###############################
        ### Convert to torch tensor ###
        ###############################
        # Use .as_tensor() instead of .tensor() to avoid a copy
        self.register_buffer('V_', torch.as_tensor(self.V, dtype=torch.float32)) # [778, 3]
        self.register_buffer('K_', torch.as_tensor(K     , dtype=torch.int64))   # [16]
        self.register_buffer('S_', torch.as_tensor(S     , dtype=torch.float32)) # [778, 3, 10]
        self.register_buffer('P_', torch.as_tensor(P     , dtype=torch.float32)) # [135, 778*3]
        self.register_buffer('J_', torch.as_tensor(J     , dtype=torch.float32)) # [16, 778]
        self.register_buffer('W_', torch.as_tensor(W     , dtype=torch.float32)) # [778, 16]
        self.register_buffer('C_', torch.as_tensor(C     , dtype=torch.float32)) # [45, 45]
        self.register_buffer('M_', torch.as_tensor(M     , dtype=torch.float32)) # [45]


        ##################
        ### MANO joint ###
        ##################
        # [-1 0 1 2 0 4 5 0 7 8  0 10 11  0 13 14] parent joint (refer to the joint labeling below)
        # [ 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15] joint order (total 16 joints including wrist)
        # MANO joint convention:
        # Note: The order of ring and little finger is swapped!
        # T   I  M  R   L (Thumb, Index, Middle, Ring, Little)
        # 16 17 18  19 20 (Fingetip newly added)
        # 15  3  6  12  9 (DIP)
        # 14  2  5  11  8 (PIP)
        # 13  1  4  10  7 (MCP)
        #       0 (Wrist)

        # Rearrange MANO joint convention to standard convention:
        #  0:Wrist,
        #  1:TMCP,  2:TPIP,  3:TDIP,  4:TTIP (Thumb)
        #  5:IMCP,  6:IPIP,  7:IDIP,  8:ITIP (Index)
        #  9:MMCP, 10:MPIP, 11:MDIP, 12:MTIP (Middle)
        # 13:RMCP, 14:RPIP, 15:RDIP, 16:RTIP (Ring)
        # 17:LMCP, 18:LPIP, 19:LDIP, 20:LTIP (Little)
        self.remap_joint = [ 0,          # Wrist
                            13,14,15,16, # Thumb
                             1, 2, 3,17, # Index
                             4, 5, 6,18, # Middle
                            10,11,12,19, # Ring
                             7, 8, 9,20] # Little

        # Vertices corresponding to fingertips
        # Thumb, Index, Middle, Ring, Little
        self.register_buffer('fingertip_vert', torch.tensor(
            [745,333,444,555,672], dtype=torch.int64)) # My own definition
            # [744,320,443,555,672], dtype=torch.int64)) # FreiHAND definition
            # [734,333,443,555,678], dtype=torch.int64)) # End-to-end Hand Mesh Recovery from a Monocular RGB Image (Zhang 2019)


        ###################################
        ### Convert joint angle to pose ###
        ###################################
        Z = self.generate_Zmat(S, self.V, J)
        self.register_buffer('Z_', torch.as_tensor(Z, dtype=torch.float32)) # [23, 45]


        #########################
        ### Define pose limit ###
        #########################
        # Values adapted from
        # HOnnotate: A method for 3D Annotation of Hand and Object Poses Supplementary Material
        # https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/team_lepetit/images/hampali/supplementary.pdf
        plim = np.array([
            # Index
            [[ 0.00, 0.45], [-0.15, 0.20], [0.10, 1.80]], # MCP
            [[-0.30, 0.20], [ 0.00, 0.00], [0.00, 0.20]], # PIP
            [[ 0.00, 0.00], [ 0.00, 0.00], [0.00, 1.25]], # DIP
            # Middle
            [[ 0.00, 0.00], [-0.15, 0.15], [0.10, 2.00]], # MCP
            [[-0.50,-0.20], [ 0.00, 0.00], [0.00, 2.00]], # PIP
            [[ 0.00, 0.00], [ 0.00, 0.00], [0.00, 1.25]], # DIP
            # Little
            [[-1.50,-0.20], [-0.15, 0.60], [-0.10, 1.60]], # MCP
            [[ 0.00, 0.00], [-0.50, 0.60], [ 0.00, 2.00]], # PIP
            [[ 0.00, 0.00], [ 0.00, 0.00], [ 0.00, 1.25]], # DIP
            # Ring
            [[-0.50,-0.40], [-0.25, 0.10], [0.10, 1.80]], # MCP
            [[-0.40,-0.20], [ 0.00, 0.00], [0.00, 2.00]], # PIP
            [[ 0.00, 0.00], [ 0.00, 0.00], [0.00, 1.25]], # DIP
            # Thumb
            [[ 0.00, 2.00], [-0.83, 0.66], [ 0.00, 0.50]], # MCP
            [[-0.15,-1.60], [ 0.00, 0.00], [ 0.00, 0.50]], # PIP
            [[ 0.00, 0.00], [-0.50, 0.00], [-1.57, 1.08]]])# DIP
        self.register_buffer('plim_', torch.as_tensor(plim, dtype=torch.float32)) # [15, 3, 2]


        ################################
        ### Define joint angle limit ###
        ################################
        alim = np.array([
            # MCP a/a, MCP f/e, PIP f/e, DIP f/e
            [-20,30], [-10,90], [-1,90], [-1,90], # Index [min,max]
            [-20,10], [-10,90], [-1,90], [-1,90], # Middle
            [-20,10], [-10,90], [-1,90], [-1,90], # Ring
            [-30,20], [-10,90], [-1,90], [-1,90], # Little
            # x-axis, y-axis  , z-axis
            # [-45,140], [-45,45], [-45,45], # Thumb TM
            [-45,45], [-45,45], [-45,45], # Thumb TM
            [-45, 45], [-45,45], [-45,45], # Thumb MCP
            [-1 , 90]])                    # Thumb IP flex/ext
        # Convert degrees to radians
        alim = np.radians(alim)
        self.register_buffer('alim_', torch.as_tensor(alim, dtype=torch.float32)) # [23, 2]

        self.ReLU = nn.ReLU() # For ReLU(x) = max(0,x)

        print('[MANO] Loaded', file_path)


    def convert_pca_to_pose(self, ppca):
        # Note: (bs, 45) dot (45, 45) = (bs, 45) + (bs, 45) = (bs, 45)
        # Then reshape to (15, 3) to be used as pose input to feedforward to model
        bs = ppca.shape[0] # Get batch size
        n = ppca.shape[1] # Get number of components used

        return (torch.matmul(ppca, self.C_[:n, :]) + self.M_).view(bs,-1,3) # [bs, 15, 3]


    def convert_ang_to_pose(self, ang):
        # Note: (bs, 23) dot (23, 45) = (bs, 45)
        # Then reshape to (15, 3) to be used as pose input to feedforward to model
        bs = ang.shape[0] # Get batch size

        return (torch.matmul(ang, self.Z_)).view(bs,-1,3) # [bs, 15, 3]


    def forward(self, beta, pose, rvec, tvec, root):
        # Combine global rotation vector (rvec) with pose to get full pose
        # [bs, 1, 3] cat [bs, 15, 3] = [bs, 16, 3]
        full_pose = torch.cat([rvec.unsqueeze(dim=1), pose], dim=1)

        # Perform linear blend skinning
        vertices, joints = lbs(beta, full_pose,
            self.V_, self.K_, self.S_, self.P_, self.J_, self.W_)

        # Add fingertip vertices to joints
        fingertips = torch.index_select(
            vertices, dim=1, index=self.fingertip_vert)
        joints = torch.cat([joints, fingertips], dim=1)

        # Rearrange joints to follow standard convention
        joints = joints[:, self.remap_joint, :]

        # Special for Freihand dataset
        joints_ = torch.matmul(self.J_, vertices)
        tvec = root - joints_[0,4,:] # Joint 3 is middle finger MCP

        # Finally apply global translation
        vertices += tvec.unsqueeze(dim=1)
        joints += tvec.unsqueeze(dim=1)

        return vertices, joints


    def forward(self, beta, pose, rvec, tvec=None):
        # Combine global rotation vector (rvec) with pose to get full pose
        # [bs, 1, 3] cat [bs, 15, 3] = [bs, 16, 3]
        full_pose = torch.cat([rvec.unsqueeze(dim=1), pose], dim=1)

        # Perform linear blend skinning
        vertices, joints = lbs(beta, full_pose,
            self.V_, self.K_, self.S_, self.P_, self.J_, self.W_)

        # Add fingertip vertices to joints
        fingertips = torch.index_select(
            vertices, dim=1, index=self.fingertip_vert)
        joints = torch.cat([joints, fingertips], dim=1)

        # Rearrange joints to follow standard convention
        joints = joints[:, self.remap_joint, :]

        if tvec is not None:
            # Finally apply global translation
            vertices += tvec.unsqueeze(dim=1)
            joints += tvec.unsqueeze(dim=1)

        return vertices, joints


    def compute_ang_limit(self, ang):
        # ReLU(x)=max(0,x)
        min_ = self.ReLU(self.alim_[:,0]-ang)
        max_ = self.ReLU(ang-self.alim_[:,1])

        return torch.sum(min_) + torch.sum(max_)


    def generate_Zmat(self, S, V, J):
        # Init Z matrix to all zeros
        Z = np.zeros((23, 45))

        # Note: MANO pose has a total of 15 joints
        #       Each joint 3 DoFs thus pose has a total 15*3 = 45 values
        # But actual human hand only has a total of 21/22/23 DoFs
        # (21 DoFs for 4 fingers(4x4) + 1 thumb(5/6/7))
        # Thus joint angle will have 23 values (using thumb with 7 DoFs)

        #######################################
        ### Get the joints at rest position ###
        #######################################
        Bs = S.dot(np.zeros(10)) # (778, 3, 10) dot (10) = (788, 3)
        Vs = V + Bs              # (788, 3) Vertices of template (V) are modified in an additive way by adding Bs
        Jrest = J.dot(Vs)        # (16, 778) dot (778, 3) = (16, 3) Rest joint locations

        ###################################
        ### Create some lamda functions ###
        ###################################
        # Convert rotation vector (3 by 1 or 1 by 3) to rotation matrix (3 by 3)
        rvec2rmat = lambda rvec : cv2.Rodrigues(rvec)[0]
        # Note: j1 is the joint of interest and j2 is its parent (1 by 3)
        rotate_finger = lambda j1, j2 : rvec2rmat(np.array(
                [0,np.arctan((j1[2]-j2[2])/(j1[0]-j2[0])),0])) # Rotate about y axis
        # Note: Thumb is rotated by around some degrees relative to the rest of the fingers
        # Note: pL is the left and pR is the right point of thumb fingernail (1 by 3)
        rotate_thumb = lambda pL, pR : rvec2rmat(np.array(
                [np.arctan((pL[1]-pR[1])/(pL[2]-pR[2])),0,0])) # Rotate about x axis

        ####################
        ### Index finger ###
        ####################
        Z[0:2,0:3] = rotate_finger(Jrest[1,:],Jrest[0,:])[1:3,:] # 0:MCP abduct/adduct, 1:MCP flex/ext
        Z[  2,3:6] = rotate_finger(Jrest[2,:],Jrest[1,:])[  2,:] # 2:PIP flex/ext
        Z[  3,6:9] = rotate_finger(Jrest[3,:],Jrest[2,:])[  2,:] # 3:DIP flex/ext
        #####################
        ### Middle finger ###
        #####################
        Z[4:6, 9:12] = rotate_finger(Jrest[4,:],Jrest[0,:])[1:3,:] # 4:MCP abduct/adduct, 5:MCP flex/ext
        Z[  6,12:15] = rotate_finger(Jrest[5,:],Jrest[4,:])[  2,:] # 6:PIP flex/ext
        Z[  7,15:18] = rotate_finger(Jrest[6,:],Jrest[5,:])[  2,:] # 7:DIP flex/ext
        ###################
        ### Ring finger ###
        ###################
        Z[8:10,27:30] = rotate_finger(Jrest[10,:],Jrest[ 0,:])[1:3,:] # 8:MCP abduct/adduct, 9:MCP flex/ext
        Z[  10,30:33] = rotate_finger(Jrest[11,:],Jrest[10,:])[  2,:] # 10:PIP flex/ext
        Z[  11,33:36] = rotate_finger(Jrest[12,:],Jrest[11,:])[  2,:] # 11:DIP flex/ext
        #####################
        ### Little finger ###
        #####################
        Z[12:14,18:21] = rotate_finger(Jrest[7,:],Jrest[0,:])[1:3,:] # 12:MCP abduct/adduct, 13:MCP flex/ext
        Z[   14,21:24] = rotate_finger(Jrest[8,:],Jrest[7,:])[  2,:] # 14:PIP flex/ext
        Z[   15,24:27] = rotate_finger(Jrest[9,:],Jrest[8,:])[  2,:] # 15:DIP flex/ext
        #############
        ### Thumb ###
        #############
        thumb_left, thumb_right = 747, 720
        Z[16:19,36:39] = np.eye(3) # 16:TM  rx, 17:TM  ry, 18:TM  rz
        Z[19:22,39:42] = np.eye(3) # 19:MCP rx, 20:MCP ry, 21:MCP rz
        Z[   22,42:45] = rotate_thumb(Vs[thumb_left,:],Vs[thumb_right,:]).dot(
            rotate_finger(Jrest[15,:],Jrest[14,:]))[2,:] # 22:IP flex/ext

        return Z


###############################################################################
### Simple example to test program                                          ###
###############################################################################
if __name__ == '__main__':
    import argparse
    import open3d as o3d

    bs = 1 # Batchsize
    beta = torch.zeros([bs,10], dtype=torch.float32)
    rvec = torch.zeros([bs,3], dtype=torch.float32)
    tvec = torch.zeros([bs,3], dtype=torch.float32)

    model = MANO()
    pose = torch.zeros([bs,15,3], dtype=torch.float32)
    ppca = torch.zeros([bs,45], dtype=torch.float32)
    # pose = model.convert_pca_to_pose(ppca)
    vertices, joints = model.forward(beta, pose, rvec, tvec)
    print('vertices', vertices.shape, vertices.dtype) # torch.Size([10, 778, 3]) torch.float32
    print('joints', joints.shape, joints.dtype)       # torch.Size([10, 21, 3]) torch.float32


    ########################################
    ### Quick visualization using Open3D ###
    ########################################
    # Create a reference frame 10 cm
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # Draw mesh model
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices[0,:,:])
    mesh.triangles = o3d.utility.Vector3iVector(model.F)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.75, 0.75, 0.75])
    # Draw wireframe
    ls = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
    ls.paint_uniform_color([0.75, 0.75, 0.75])
    # Draw joints
    mesh_spheres = []
    for j in joints[0,:,:]:
        m = o3d.geometry.TriangleMesh.create_sphere(radius=0.001)
        m.compute_vertex_normals()
        m.paint_uniform_color([1,0,0])
        m.translate(j)
        mesh_spheres.append(m)

    o3d.visualization.draw_geometries([mesh, mesh_frame])
    o3d.visualization.draw_geometries([ls, mesh_frame] + mesh_spheres)