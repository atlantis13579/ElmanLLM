// conv2d_input_grad scalar implementation.
int input_width=P(5);int input_height=P(4);int input_channels=P(3);int ix=i%input_width;int iy=(i/input_width)%input_height;int ic=(i/(input_width*input_height))%input_channels;int batch=i/(input_channels*input_height*input_width);
int output_channels=P(12);int output_height=P(13);int output_width=P(14);int kernel_height=P(30);int kernel_width=P(31);int channels_per_group=P(29);int outputs_per_group=output_channels/P(70);int group=ic/channels_per_group;int local_channel=ic%channels_per_group;float value=0.0f;
for(int local_output=0;local_output<outputs_per_group;++local_output)for(int ky=0;ky<kernel_height;++ky)for(int kx=0;kx<kernel_width;++kx){
    int numerator_y=iy+P(66)-ky*P(68);int numerator_x=ix+P(67)-kx*P(69);
    if(numerator_y>=0&&numerator_x>=0&&numerator_y%P(64)==0&&numerator_x%P(65)==0){int oy=numerator_y/P(64);int ox=numerator_x/P(65);int oc=group*outputs_per_group+local_output;
        if(oy<output_height&&ox<output_width)value+=A(((batch*output_channels+oc)*output_height+oy)*output_width+ox)*B(((oc*channels_per_group+local_channel)*kernel_height+ky)*kernel_width+kx);
    }
}
W(i,value);
