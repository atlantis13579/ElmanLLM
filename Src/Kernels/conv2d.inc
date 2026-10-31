// conv2d scalar implementation.
int output_width=P(5);int output_height=P(4);int output_channels=P(3);
int ox=i%output_width;int oy=(i/output_width)%output_height;int oc=(i/(output_width*output_height))%output_channels;int batch=i/(output_channels*output_height*output_width);
int input_channels=P(12);int input_height=P(13);int input_width=P(14);int kernel_height=P(30);int kernel_width=P(31);int channels_per_group=P(29);int outputs_per_group=output_channels/P(70);int first_channel=(oc/outputs_per_group)*channels_per_group;
float value=P(44)>=0?C(oc):0.0f;
for(int ic=0;ic<channels_per_group;++ic)for(int ky=0;ky<kernel_height;++ky)for(int kx=0;kx<kernel_width;++kx){
    int iy=oy*P(64)-P(66)+ky*P(68);int ix=ox*P(65)-P(67)+kx*P(69);
    if(iy>=0&&iy<input_height&&ix>=0&&ix<input_width)value+=A(((batch*input_channels+first_channel+ic)*input_height+iy)*input_width+ix)*B(((oc*channels_per_group+ic)*kernel_height+ky)*kernel_width+kx);
}
W(i,value);
