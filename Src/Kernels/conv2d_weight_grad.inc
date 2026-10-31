// conv2d_weight_grad scalar implementation.
int kernel_width=P(5);int kernel_height=P(4);int channels_per_group=P(3);int output_channels=P(2);int kx=i%kernel_width;int ky=(i/kernel_width)%kernel_height;int local_channel=(i/(kernel_width*kernel_height))%channels_per_group;int oc=i/(channels_per_group*kernel_height*kernel_width);
int batches=P(11);int input_channels=P(12);int input_height=P(13);int input_width=P(14);int output_height=P(30);int output_width=P(31);int outputs_per_group=output_channels/P(70);int ic=(oc/outputs_per_group)*channels_per_group+local_channel;float value=0.0f;
for(int batch=0;batch<batches;++batch)for(int oy=0;oy<output_height;++oy)for(int ox=0;ox<output_width;++ox){int iy=oy*P(64)-P(66)+ky*P(68);int ix=ox*P(65)-P(67)+kx*P(69);
    if(iy>=0&&iy<input_height&&ix>=0&&ix<input_width)value+=A(((batch*input_channels+ic)*input_height+iy)*input_width+ix)*B(((batch*output_channels+oc)*output_height+oy)*output_width+ox);
}
W(i,value);
