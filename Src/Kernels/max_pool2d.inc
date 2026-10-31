// max_pool2d scalar implementation.
int output_width=P(5);int output_height=P(4);int plane=i/(output_width*output_height);int oy=(i/output_width)%output_height;int ox=i%output_width;
int input_height=P(13);int input_width=P(14);float value=-EL_INF;int count=0;
for(int ky=0;ky<P(64);++ky)for(int kx=0;kx<P(65);++kx){int iy=oy*P(66)-P(68)+ky;int ix=ox*P(67)-P(69)+kx;
    if(iy>=0&&iy<input_height&&ix>=0&&ix<input_width){float x=A((plane*input_height+iy)*input_width+ix);value=EL_MAX(value,x);++count;}
    else if(P(70)!=0)++count;
}
W(i,value);
