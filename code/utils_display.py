###############################################################################
### Useful function for visualizing results
###############################################################################

import cv2
import torch
import numpy as np
import open3d as o3d


class Display:
    def __init__(self, arg, model, device):
        self.arg = arg
        self.model = model
        self.device = device

        # Draw hand mesh
        self.create_hand_mesh()

        # Draw hand skeleton
        self.create_hand_skeleton()

        # Open3D visualization
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=224, height=224)

        # Add geometry
        self.vis.add_geometry(self.mesh)
        self.vis.add_geometry(self.bone)
        self.vis.add_geometry(self.joint)

        # Set camera view
        ctr = self.vis.get_view_control()
        ctr.set_up([0,-1,0])    # Set up as -y axis
        ctr.set_front([0,0,-1]) # Set to looking towards -z axis
        ctr.set_lookat([0,0,0]) # Set to center of view
        ctr.set_zoom(1.2)       # Enlarge slightly

        # # Generate video for documentation define codec and create VideoWriter object
        # fps = 30
        # width, height = 224*3, 224
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
        # self.video = cv2.VideoWriter('../data/video_result.mp4', fourcc, fps, (width, height))


    def update(self, image, result, time=None):
        # Unpack result and detach from PyTorch graph
        keypt, joint, vert, pose, param = result
        keypt = keypt[0].cpu().detach()
        joint = joint[0].cpu().detach()
        vert  = vert[0].cpu().detach()
        scale = param[:,0].cpu().detach()
        trans = param[:,1:3].cpu().detach()

        # Convert image tensor to numpy for OpenCV
        image = (image.permute(1,2,0) * 255).contiguous().numpy().astype(np.uint8)
        ori = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        self.draw_mesh(img, vert, self.model.mano.F, scale, trans) # Note: Quite time consuming
        self.draw_skeleton(img, keypt)

        if time is not None:
            cv2.putText(ori, 'FPS: %.0f' % (1 / time),
                (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.imshow('img', np.hstack((ori, img)))

        # Centralize vert and joint relative to middle finger MCP joint [9]
        vert  = vert - joint[9]
        joint = joint - joint[9]

        # Update Open3D mesh, bone and joint
        self.mesh.vertices = o3d.utility.Vector3dVector(vert)
        self.bone.points   = o3d.utility.Vector3dVector(joint)
        self.joint.points  = o3d.utility.Vector3dVector(joint)

        self.vis.update_geometry(None)
        self.vis.poll_events()
        self.vis.update_renderer()

        # # Generate image / video for documentation
        # mesh = (np.asarray(self.vis.capture_screen_float_buffer())*255).astype(np.uint8)
        # mesh = cv2.cvtColor(mesh, cv2.COLOR_RGB2BGR)
        # comb = np.hstack((ori, img, mesh))
        # # cv2.imwrite('../data/' + self.arg.data + '_result.png', comb)
        # self.video.write(comb)


    def create_hand_mesh(self):
        rvec = torch.zeros((1,3),  dtype=torch.float32, device=self.device)
        beta = torch.zeros((1,10), dtype=torch.float32, device=self.device)
        ppca = torch.zeros((1,10), dtype=torch.float32, device=self.device)
        pose = self.model.mano.convert_pca_to_pose(ppca)

        vert, joint = self.model.mano.forward(beta, pose, rvec) # Feedforward to get initial output

        # Convert from m to mm
        vert *= 1000.0
        joint *= 1000.0

        self.mesh = o3d.geometry.TriangleMesh()
        self.mesh.vertices = o3d.utility.Vector3dVector(vert[0,:,:].cpu().detach())
        self.mesh.triangles = o3d.utility.Vector3iVector(self.model.mano.F)
        self.mesh.paint_uniform_color([255/255, 172/255, 150/255]) # Skin color
        self.mesh.compute_vertex_normals()


    def create_hand_skeleton(self):
        joints = np.zeros((21,3))

        # Draw the 20 bones (lines) connecting 21 joints
        # The lines below is the kinematic tree that defines the connection between parent and child joints
        # Below lists the definition of the 21 joints following standard convention
        #  0:Wrist,
        #  1:TMCP, 2:TPIP, 3:TDIP, 4:TTIP (Thumb)
        #  5:IMCP, 6:IPIP, 7:IDIP, 8:ITIP (Index)
        #  9:MMCP,10:MPIP,11:MDIP,12:MTIP (Middle)
        # 13:RMCP,14:RPIP,15:RDIP,16:RTIP (Ring)
        # 17:PMCP,18:PPIP,19:PDIP,20:PTIP (Little)
        lines = [[0,1],  [1,2],   [2,3],  [3,4],
                 [0,5],  [5,6],   [6,7],  [7,8],
                 [0,9], [9,10], [10,11],[11,12],
                 [0,13],[13,14],[14,15],[15,16],
                 [0,17],[17,18],[18,19],[19,20]]
        colors = [[255, 0, 0], [255, 60, 0], [255, 120, 0], [255, 180, 0], # Thumb red
                  [0, 255, 0], [60, 255, 0], [120, 255, 0], [180, 255, 0], # Index green yellow
                  [0, 255, 0], [0, 255, 60], [0, 255, 120], [0, 255, 180], # Middle green
                  [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255], # Ring blue cyan
                  [0, 0, 255], [60, 0, 255], [120, 0, 255], [180, 0, 255]] # Little blue magenta
        colors = np.array(colors) / 255

        self.bone = o3d.geometry.LineSet()
        self.bone.lines  = o3d.utility.Vector2iVector(lines)
        self.bone.points = o3d.utility.Vector3dVector(joints)
        self.bone.colors = o3d.utility.Vector3dVector(colors)

        # Draw the 21 joints (pcd)
        colors = [[0,0,0], # Black wrist
                  [255, 0, 0], [255, 60, 0], [255, 120, 0], [255, 180, 0], # Thumb blue
                  [0, 255, 0], [60, 255, 0], [120, 255, 0], [180, 255, 0], # Index green blue
                  [0, 255, 0], [0, 255, 60], [0, 255, 120], [0, 255, 180], # Middle green red
                  [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255], # Ring red green
                  [0, 0, 255], [60, 0, 255], [120, 0, 255], [180, 0, 255]] # Little red blue
        colors = np.array(colors) / 255

        self.joint = o3d.geometry.PointCloud()
        self.joint.points = o3d.utility.Vector3dVector(joints)
        self.joint.colors = o3d.utility.Vector3dVector(colors)


    def draw_mesh(self, img, vert, face, scale, trans, color=(200,200,200)):
        # Project 3D vertices to 2D points
        # Add homo coor verts
        # matrix = np.array([
        #     [607.92271,         0, 314.78337],
        #     [        0, 607.88192, 236.42484],
        #     [        0,         0,         1]])
        # verts = np.hstack((verts, np.ones((verts.shape[0], 1))))
        # # Add empty translation to intrinsic camera matrix
        # matrix = np.hstack((matrix, np.zeros((3,1))))
        # uv = np.matmul(matrix, verts.T).T # matrix (3,4), verts (n,4) -> uv (n, 4)
        # uv = uv[:,:3]
        # uv = uv[:, :2] / uv[:, -1:] # (n, 2)
        # uv = uv.astype(np.int32) # Convert to integer

        # Weak perspective projection
        uv = vert[:,:2] * scale + trans

        # Draw the lines connecting the 2D points
        for f in face:
            u0, v0 = uv[f[0],:]
            u1, v1 = uv[f[1],:]
            u2, v2 = uv[f[2],:]
            # p0 = (u0,v0)
            # p1 = (u1,v1)
            # p2 = (u2,v2)
            # pp = np.array([p0,p1,p2])
            # cv2.drawContours(img, [pp], 0, (0,0,255), -1)
            cv2.line(img, (u0,v0), (u1, v1), color, 1)
            cv2.line(img, (u1,v1), (u2, v2), color, 1)
            cv2.line(img, (u2,v2), (u0, v0), color, 1)

        return img


    def draw_skeleton(self, img, keypt):
        # Define color for joint
        color = [[0,0,0], # Wrist
                 [255,0,0],[255,60,0],[255,120,0],[255,180,0], # Thumb
                 [0,255,0],[60,255,0],[120,255,0],[180,255,0], # Index
                 [0,255,0],[0,255,60],[0,255,120],[0,255,180], # Middle
                 [0,0,255],[0,60,255],[0,120,255],[0,180,255], # Ring
                 [0,0,255],[60,0,255],[120,0,255],[180,0,255]] # Little

        # Define kinematic tree to link keypt together to form skeleton
        ktree = [0,          # Wrist
                 0,1,2,3,    # Thumb
                 0,5,6,7,    # Index
                 0,9,10,11,  # Middle
                 0,13,14,15, # Ring
                 0,17,18,19] # Little

        for i, (x, y) in enumerate(keypt):
            # Draw joint
            cv2.circle(img, (x, y), 2, color[i], -1)
            # Draw bone
            x0, y0 = keypt[ktree[i],:]
            cv2.line(img, (x0, y0), (x, y), color[i], 1)