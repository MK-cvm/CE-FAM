import torch
import torch.nn as nn

class MapConverter(nn.Module):
    def __init__(self, concept_dim=768, num_classes=1000, concept_num=1197, layer_dim=(256,512,1024,2048), scale=0, bias=0, model_type="siglip", single_layer=False):
        super(MapConverter, self).__init__()
        self.inplanes = layer_dim
        self.num_classes = num_classes
        self.concept_dim = concept_dim
        self.concept_num = concept_num
        self.layer_dim4 = layer_dim[3]
        self.layer_dim3 = layer_dim[2]
        self.layer_dim2 = layer_dim[1]
        self.layer_dim1 = layer_dim[0]
        self.single_layer = single_layer
        if self.single_layer:
            self.fc1 = nn.Linear(self.layer_dim4, 1024)
            self.fc2 = nn.Linear(1024, 1024)
        else:
            if self.layer_dim4 > 1024:
                self.fc1 = nn.Linear(self.layer_dim4+self.layer_dim3+self.layer_dim2+self.layer_dim1, 2048)
                self.fc2 = nn.Linear(2048, 1024)
            else:
                self.fc1 = nn.Linear(self.layer_dim4+self.layer_dim3+self.layer_dim2+self.layer_dim1, 1024)
                self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, self.concept_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.sigmoid = nn.Sigmoid()
        self.scale = scale
        self.bias = bias
        self.model_type = model_type
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, cnn_layer_vector, concept_vectors=None, important_index=None, layer_ind=None):
        if len(cnn_layer_vector) == 4:
            cnn_layer_vector4 = cnn_layer_vector[3][:, :, 0, 0]
            cnn_layer_vector3 = cnn_layer_vector[2][:, :, 0, 0]
            cnn_layer_vector2 = cnn_layer_vector[1][:, :, 0, 0]
            cnn_layer_vector1 = cnn_layer_vector[0][:, :, 0, 0]
        else:
            cnn_layer_vector4 = cnn_layer_vector[:, :, 0, 0]
        if layer_ind is not None:
            if layer_ind == 0:
                avg_value = torch.mean(cnn_layer_vector1)
                cnn_layer_vector1[0, important_index] = avg_value
            elif layer_ind == 1:
                avg_value = torch.mean(cnn_layer_vector2)
                cnn_layer_vector2[0, important_index] = avg_value
            elif layer_ind == 2:
                avg_value = torch.mean(cnn_layer_vector3)
                cnn_layer_vector3[0, important_index] = avg_value
            elif layer_ind == 3:
                avg_value = torch.mean(cnn_layer_vector4)
                cnn_layer_vector4[0, important_index] = avg_value
        if self.single_layer:
            x = cnn_layer_vector4
        else:
            x = torch.cat([cnn_layer_vector4, cnn_layer_vector3, cnn_layer_vector2, cnn_layer_vector1], dim=-1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        cnn_layer_embeds = self.fc3(x)
        clip_space_embeds_ = cnn_layer_embeds / cnn_layer_embeds.norm(p=2, dim=-1, keepdim=True)
        layer_sims = torch.bmm(concept_vectors, clip_space_embeds_.unsqueeze(2)) * self.scale + self.bias
        concept_pred_similarity = layer_sims[:, :, 0]
        return cnn_layer_embeds, concept_pred_similarity