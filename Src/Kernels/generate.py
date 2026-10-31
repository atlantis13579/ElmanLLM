"""Regenerate the checked-in, dependency-free native operator wrappers."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "Src" / "Kernels"
OPERATORS = ROOT / "Src" / "Operators"
bodies = {}

def write(path, text, encoding="utf-8"):
    """Preserve timestamps when regeneration does not change an operator."""
    if not path.exists() or path.read_text(encoding=encoding) != text:
        path.write_text(text, encoding=encoding, newline="\n")

binary = {
    "add": "x+y", "sub": "x-y", "mul": "x*y", "div": "x/y",
    "pow": "EL_POW(x,y)", "maximum": "EL_MAX(x,y)", "minimum": "EL_MIN(x,y)",
    "eq": "float(x==y)", "ne": "float(x!=y)", "gt": "float(x>y)",
    "ge": "float(x>=y)", "lt": "float(x<y)", "le": "float(x<=y)",
    "logical_and": "float(x!=0.0f && y!=0.0f)",
    "logical_or": "float(x!=0.0f || y!=0.0f)",
}
for name, expression in binary.items():
    bodies[name] = f"float x=A(EL_BINDEX(i,10,11,19));\nfloat y=B(EL_BINDEX(i,27,28,36));\nW(i,{expression});\n"

unary = {
    "sqrt": "EL_SQRT(x)", "rsqrt": "1.0f/EL_SQRT(x)", "exp": "EL_EXP(x)",
    "log": "EL_LOG(x)", "sin": "EL_SIN(x)", "cos": "EL_COS(x)",
    "abs": "EL_ABS(x)", "sign": "float(x>0.0f)-float(x<0.0f)",
    "logical_not": "float(x==0.0f)", "relu": "EL_MAX(x,0.0f)",
    "silu": "x/(1.0f+EL_EXP(-x))", "swish": "x/(1.0f+EL_EXP(-x))",
    "gelu": "0.5f*x*(1.0f+EL_TANH(0.7978845608028654f*(x+0.044715f*x*x*x)))",
    "leaky_relu": "x>=0.0f?x:F(64)*x", "cast": "x",
}
for name, expression in unary.items():
    bodies[name] = f"float x=A(i);\nW(i,{expression});\n"

for name in ["reshape", "squeeze", "unsqueeze", "copy"]:
    bodies[name] = "COPY_A(i,i);\n"

for name in ["sum", "mean", "max", "min", "amax", "amin", "prod"]:
    initial = "-EL_INF" if name in ("max", "amax") else "EL_INF" if name in ("min", "amin") else "1.0f" if name == "prod" else "0.0f"
    expression = "EL_MAX(value,x)" if name in ("max", "amax") else "EL_MIN(value,x)" if name in ("min", "amin") else "value*x" if name == "prod" else "value+x"
    bodies[name] = f"""int axis=P(64);
int inner=axis<0?1:P(19+axis);
int length=axis<0?EL_COUNT(10,11):P(11+axis);
int base=axis<0?0:(i/inner)*length*inner+i%inner;
float value={initial};
for(int j=0;j<length;++j){{float x=A(base+j*inner);value={expression};}}
W(i,{"value/float(length)" if name == "mean" else "value"});
"""

bodies["reduce_grad"] = """int axis=P(64);
int inner=axis<0?1:P(19+axis);
int length=axis<0?P(0):P(11+axis);
int target=axis<0?0:(i/(length*inner))*inner+i%inner;
float value=C(target);
int kind=P(66);
if(kind==1)value/=float(length);
if(kind==2||kind==3){
    float extremum=B(target);
    int base=axis<0?0:(target/inner)*length*inner+target%inner;
    int matches=0;
    for(int j=0;j<length;++j)if(A(base+j*inner)==extremum)++matches;
    value=A(i)==extremum?value/float(matches):0.0f;
}
if(kind==4){
    int base=axis<0?0:(target/inner)*length*inner+target%inner;
    float product=1.0f;
    for(int j=0;j<length;++j)if(base+j*inner!=i)product*=A(base+j*inner);
    value*=product;
}
W(i,value);
"""
bodies["permute"] = """int flat=i;int source=0;
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);source+=coordinate*P(19+P(64+axis));}
COPY_A(i,source);
"""
bodies["transpose"] = """int flat=i;int source=0;
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);int source_axis=axis==P(64)?P(65):(axis==P(65)?P(64):axis);source+=coordinate*P(19+source_axis);}
COPY_A(i,source);
"""
bodies["slice"] = """int flat=i;int source=0;
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);if(axis==P(64))coordinate=P(65)+coordinate*P(66);source+=coordinate*P(19+axis);}
COPY_A(i,source);
"""
bodies["split"] = bodies["slice"]
bodies["slice_grad"] = """int flat=i;int source=0;int valid=1;
for(int axis=P(1)-1;axis>=0;--axis){
    int coordinate=flat%P(2+axis);flat/=P(2+axis);
    if(axis==P(64)){coordinate-=P(65);if(coordinate<0||coordinate%P(66)!=0)valid=0;coordinate/=P(66);if(coordinate>=P(11+axis))valid=0;}
    source+=coordinate*P(19+axis);
}
W(i,valid!=0?A(source):0.0f);
"""
bodies["gather"] = """int flat=i;int source=0;int selected=IB(i);int valid=selected>=0&&selected<P(11+P(64));
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);if(axis==P(64))coordinate=selected;source+=coordinate*P(19+axis);}
if(valid!=0){COPY_A(i,source);}else{W(i,0.0f);}
"""
bodies["index_select"] = """int axis=P(64);int inner=P(19+axis);int output_length=P(2+axis);
int selected=IB((i/inner)%output_length);
int source=(i/(inner*output_length))*P(11+axis)*inner+selected*inner+i%inner;
if(selected>=0&&selected<P(11+axis)){COPY_A(i,source);}else{W(i,0.0f);}
"""
bodies["gather_grad"] = """float value=0.0f;
int count=EL_COUNT(10,11);
for(int j=0;j<count;++j){int flat=j;int target=0;int stride=1;int selected=IB(j);
    for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(11+axis);flat/=P(11+axis);if(axis==P(64))coordinate=selected;target+=coordinate*stride;stride*=P(2+axis);}
    if(target==i&&selected>=0&&selected<P(2+P(64)))value+=A(j);
}
W(i,value);
"""
bodies["scatter_source_grad"] = """int flat=i;int target=0;int selected=IB(i);int valid=selected>=0&&selected<P(11+P(64));
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);if(axis==P(64))coordinate=selected;target+=coordinate*P(19+axis);}
for(int j=i+1;j<P(0);++j){int other_flat=j;int other_target=0;int other_selected=IB(j);
    for(int axis=P(1)-1;axis>=0;--axis){int coordinate=other_flat%P(2+axis);other_flat/=P(2+axis);if(axis==P(64))coordinate=other_selected;other_target+=coordinate*P(19+axis);}
    if(other_target==target&&other_selected>=0&&other_selected<P(11+P(64)))valid=0;
}
W(i,valid!=0?A(target):0.0f);
"""
bodies["index_select_grad"] = """int axis=P(64);int inner=EL_OSTRIDE(axis);int length=P(2+axis);int selected=(i/inner)%length;
int count=EL_COUNT(27,28);float value=0.0f;
for(int j=0;j<count;++j)if(IB(j)==selected)value+=A((i/(length*inner))*count*inner+j*inner+i%inner);
W(i,value);
"""
for name in ["scatter", "scatter_add"]:
    bodies[name] = """float value=A(i);int integer_value=IA(i);
int count=EL_COUNT(44,45);
for(int j=0;j<count;++j){int flat=j;int target=0;int selected=IB(j);
    for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(45+axis);flat/=P(45+axis);if(axis==P(64))coordinate=selected;target+=coordinate*P(19+axis);}
    if(target==i&&selected>=0&&selected<P(2+P(64))){""" + ("value+=C(j);integer_value+=IC(j);" if name == "scatter_add" else "value=C(j);integer_value=IC(j);") + """}
}
if(P(255)==0)W(i,value);else WI(i,integer_value);
"""
bodies["concat"] = """int axis=P(64);int inner=P(19+axis);int full=P(2+axis);int coordinate=(i/inner)%full;int outer=i/(inner*full);int rest=i%inner;
if(coordinate<P(11+axis)){int source=outer*P(11+axis)*inner+coordinate*inner+rest;COPY_A(i,source);}
else{int source=outer*P(28+axis)*inner+(coordinate-P(11+axis))*inner+rest;if(P(253)==0)W(i,B(source));else WI(i,IB(source));}
"""
bodies["repeat"] = """int flat=i;int source=0;int delta=P(1)-P(10);
for(int axis=P(1)-1;axis>=0;--axis){int coordinate=flat%P(2+axis);flat/=P(2+axis);int source_axis=axis-delta;if(source_axis>=0)source+=(coordinate%P(11+source_axis))*P(19+source_axis);}
COPY_A(i,source);
"""
bodies["expand"] = "COPY_A(i,EL_BINDEX(i,10,11,19));\n"
bodies["repeat_grad"] = """float value=0.0f;int count=EL_COUNT(10,11);int delta=P(10)-P(1);
for(int j=0;j<count;++j){int flat=j;int target=0;int stride=1;
    for(int axis=P(10)-1;axis>=0;--axis){int coordinate=flat%P(11+axis);flat/=P(11+axis);int target_axis=axis-delta;if(target_axis>=0){target+=(coordinate%P(2+target_axis))*stride;stride*=P(2+target_axis);}}
    if(target==i)value+=A(j);
}
W(i,value);
"""
bodies["uniform"] = "float random_value=elman_random(i,P(64),P(65));\nW(i,F(66)+(F(67)-F(66))*random_value);\n"
bodies["normal"] = "float u=elman_random(2*i,P(64),P(65));\nfloat v=elman_random(2*i+1,P(64),P(65));\nW(i,F(66)+F(67)*EL_SQRT(-2.0f*EL_LOG(u))*EL_COS(6.283185307179586f*v));\n"
bodies["dropout"] = "float value=A(i);\nif(P(67)!=0){float probability=F(66);value=probability>=1.0f?0.0f:(elman_random(i,P(64),P(65))>=probability?value/(1.0f-probability):0.0f);}\nW(i,value);\n"

matrix = """int rows=P(2+P(1)-2);int columns=P(2+P(1)-1);int row=(i/columns)%rows;int column=i%columns;int batch=i/(rows*columns);
int abase=EL_MATRIX_BATCH(batch,10,11,19);int bbase=EL_MATRIX_BATCH(batch,27,28,36);
int shared_size=P(11+P(10)-1);float value=0.0f;
for(int k=0;k<shared_size;++k)value+=A(abase+row*shared_size+k)*B(bbase+k*columns+column);
"""
bodies["matmul"] = matrix + "W(i,value);\n"
bodies["bmm"] = bodies["matmul"]
bodies["addmm"] = matrix + "W(i,F(64)*value+F(65)*C(EL_BINDEX(i,44,45,53)));\n"
for name in ["triu", "tril"]:
    comparison = ">=" if name == "triu" else "<="
    bodies[name] = f"int columns=P(2+P(1)-1);int rows=P(2+P(1)-2);int row=(i/columns)%rows;int column=i%columns;\nif(column-row{comparison}P(64)){{COPY_A(i,i);}}else{{W(i,0.0f);}}\n"

softmax = """int axis=P(64);int inner=P(19+axis);int length=P(11+axis);int base=(i/(length*inner))*length*inner+i%inner;
float maximum_value=-EL_INF;for(int j=0;j<length;++j)maximum_value=EL_MAX(maximum_value,A(base+j*inner));
float denominator=0.0f;for(int j=0;j<length;++j)denominator+=EL_EXP(A(base+j*inner)-maximum_value);
"""
bodies["softmax"] = softmax + "W(i,EL_EXP(A(i)-maximum_value)/denominator);\n"
bodies["log_softmax"] = softmax + "W(i,A(i)-maximum_value-EL_LOG(denominator));\n"
for name in ["layer_norm", "rms_norm"]:
    bodies[name] = """int length=1;for(int axis=P(10)-P(64);axis<P(10);++axis)length*=P(11+axis);
int base=(i/length)*length;float average=0.0f;
""" + ("for(int j=0;j<length;++j)average+=A(base+j);average/=float(length);\n" if name == "layer_norm" else "") + """float variance=0.0f;for(int j=0;j<length;++j){float difference=A(base+j)-average;variance+=difference*difference;}variance/=float(length);
float value=(A(i)-average)/EL_SQRT(variance+F(65));int coordinate=i%length;
if(P(27)>=0)value*=B(coordinate);if(P(44)>=0)value+=C(coordinate);
W(i,value);
"""
bodies["batch_norm"] = """int channels=P(12);int spatial=P(20);int channel=(i/spatial)%channels;
W(i,(A(i)-B(channel))/EL_SQRT(C(channel)+F(64)));
"""
bodies["group_norm"] = """int channels=P(12);int spatial=P(20);int per_group=channels/P(64);int length=per_group*spatial;int base=(i/length)*length;
float average=0.0f;for(int j=0;j<length;++j)average+=A(base+j);average/=float(length);
float variance=0.0f;for(int j=0;j<length;++j){float difference=A(base+j)-average;variance+=difference*difference;}variance/=float(length);
float value=(A(i)-average)/EL_SQRT(variance+F(65));int channel=(i/spatial)%channels;
if(P(27)>=0)value*=B(channel);if(P(44)>=0)value+=C(channel);W(i,value);
"""
bodies["cross_entropy"] = """int rows=P(11);int columns=P(12);float total=0.0f;
for(int row=0;row<rows;++row){float maximum_value=-EL_INF;for(int j=0;j<columns;++j)maximum_value=EL_MAX(maximum_value,A(row*columns+j));
    float denominator=0.0f;for(int j=0;j<columns;++j)denominator+=EL_EXP(A(row*columns+j)-maximum_value);
    int target=IB(row);if(target>=0&&target<columns)total+=EL_LOG(denominator)+maximum_value-A(row*columns+target);else total+=EL_INF;
}
W(i,total/float(rows));
"""
bodies["cross_entropy_grad"] = """int rows=P(11);int columns=P(12);int row=i/columns;int column=i%columns;float maximum_value=-EL_INF;
for(int j=0;j<columns;++j)maximum_value=EL_MAX(maximum_value,A(row*columns+j));
float denominator=0.0f;for(int j=0;j<columns;++j)denominator+=EL_EXP(A(row*columns+j)-maximum_value);
float value=EL_EXP(A(i)-maximum_value)/denominator-float(column==IB(row));W(i,value*C(0)/float(rows));
"""
bodies["mse_loss"] = """int count=EL_COUNT(10,11);float total=0.0f;
for(int j=0;j<count;++j){float difference=A(j)-B(j);total+=difference*difference;}
W(i,total/float(count));
"""

bodies["conv2d"] = """int output_width=P(5);int output_height=P(4);int output_channels=P(3);
int ox=i%output_width;int oy=(i/output_width)%output_height;int oc=(i/(output_width*output_height))%output_channels;int batch=i/(output_channels*output_height*output_width);
int input_channels=P(12);int input_height=P(13);int input_width=P(14);int kernel_height=P(30);int kernel_width=P(31);int channels_per_group=P(29);int outputs_per_group=output_channels/P(70);int first_channel=(oc/outputs_per_group)*channels_per_group;
float value=P(44)>=0?C(oc):0.0f;
for(int ic=0;ic<channels_per_group;++ic)for(int ky=0;ky<kernel_height;++ky)for(int kx=0;kx<kernel_width;++kx){
    int iy=oy*P(64)-P(66)+ky*P(68);int ix=ox*P(65)-P(67)+kx*P(69);
    if(iy>=0&&iy<input_height&&ix>=0&&ix<input_width)value+=A(((batch*input_channels+first_channel+ic)*input_height+iy)*input_width+ix)*B(((oc*channels_per_group+ic)*kernel_height+ky)*kernel_width+kx);
}
W(i,value);
"""
bodies["conv2d_input_grad"] = """int input_width=P(5);int input_height=P(4);int input_channels=P(3);int ix=i%input_width;int iy=(i/input_width)%input_height;int ic=(i/(input_width*input_height))%input_channels;int batch=i/(input_channels*input_height*input_width);
int output_channels=P(12);int output_height=P(13);int output_width=P(14);int kernel_height=P(30);int kernel_width=P(31);int channels_per_group=P(29);int outputs_per_group=output_channels/P(70);int group=ic/channels_per_group;int local_channel=ic%channels_per_group;float value=0.0f;
for(int local_output=0;local_output<outputs_per_group;++local_output)for(int ky=0;ky<kernel_height;++ky)for(int kx=0;kx<kernel_width;++kx){
    int numerator_y=iy+P(66)-ky*P(68);int numerator_x=ix+P(67)-kx*P(69);
    if(numerator_y>=0&&numerator_x>=0&&numerator_y%P(64)==0&&numerator_x%P(65)==0){int oy=numerator_y/P(64);int ox=numerator_x/P(65);int oc=group*outputs_per_group+local_output;
        if(oy<output_height&&ox<output_width)value+=A(((batch*output_channels+oc)*output_height+oy)*output_width+ox)*B(((oc*channels_per_group+local_channel)*kernel_height+ky)*kernel_width+kx);
    }
}
W(i,value);
"""
bodies["conv2d_weight_grad"] = """int kernel_width=P(5);int kernel_height=P(4);int channels_per_group=P(3);int output_channels=P(2);int kx=i%kernel_width;int ky=(i/kernel_width)%kernel_height;int local_channel=(i/(kernel_width*kernel_height))%channels_per_group;int oc=i/(channels_per_group*kernel_height*kernel_width);
int batches=P(11);int input_channels=P(12);int input_height=P(13);int input_width=P(14);int output_height=P(30);int output_width=P(31);int outputs_per_group=output_channels/P(70);int ic=(oc/outputs_per_group)*channels_per_group+local_channel;float value=0.0f;
for(int batch=0;batch<batches;++batch)for(int oy=0;oy<output_height;++oy)for(int ox=0;ox<output_width;++ox){int iy=oy*P(64)-P(66)+ky*P(68);int ix=ox*P(65)-P(67)+kx*P(69);
    if(iy>=0&&iy<input_height&&ix>=0&&ix<input_width)value+=A(((batch*input_channels+ic)*input_height+iy)*input_width+ix)*B(((batch*output_channels+oc)*output_height+oy)*output_width+ox);
}
W(i,value);
"""
for name in ["max_pool2d", "avg_pool2d"]:
    is_max = name.startswith("max")
    bodies[name] = """int output_width=P(5);int output_height=P(4);int plane=i/(output_width*output_height);int oy=(i/output_width)%output_height;int ox=i%output_width;
int input_height=P(13);int input_width=P(14);float value=""" + ("-EL_INF" if is_max else "0.0f") + """;int count=0;
for(int ky=0;ky<P(64);++ky)for(int kx=0;kx<P(65);++kx){int iy=oy*P(66)-P(68)+ky;int ix=ox*P(67)-P(69)+kx;
    if(iy>=0&&iy<input_height&&ix>=0&&ix<input_width){float x=A((plane*input_height+iy)*input_width+ix);""" + ("value=EL_MAX(value,x);" if is_max else "value+=x;") + """++count;}
    else if(P(70)!=0)++count;
}
W(i,""" + ("value" if is_max else "count>0?value/float(count):0.0f") + ");\n"

for name in ["max_pool2d_grad", "avg_pool2d_grad"]:
    is_max = name.startswith("max")
    bodies[name] = """int input_width=P(5);int input_height=P(4);int plane=i/(input_width*input_height);int iy=(i/input_width)%input_height;int ix=i%input_width;int output_height=P(30);int output_width=P(31);float value=0.0f;
for(int oy=0;oy<output_height;++oy)for(int ox=0;ox<output_width;++ox){int start_y=oy*P(66)-P(68);int start_x=ox*P(67)-P(69);
    if(iy>=start_y&&iy<start_y+P(64)&&ix>=start_x&&ix<start_x+P(65)){
""" + ("""        float maximum_value=-EL_INF;int selected=-1;
        for(int ky=0;ky<P(64);++ky)for(int kx=0;kx<P(65);++kx){int y=start_y+ky;int x=start_x+kx;if(y>=0&&y<input_height&&x>=0&&x<input_width){int index=(plane*input_height+y)*input_width+x;float candidate=A(index);if(selected<0||candidate>maximum_value){maximum_value=candidate;selected=index;}}}
        if(selected==i)value+=B((plane*output_height+oy)*output_width+ox);
""" if is_max else """        int count=P(64)*P(65);
        if(P(70)==0){count=0;for(int ky=0;ky<P(64);++ky)for(int kx=0;kx<P(65);++kx){int y=start_y+ky;int x=start_x+kx;if(y>=0&&y<input_height&&x>=0&&x<input_width)++count;}}
        if(count>0)value+=B((plane*output_height+oy)*output_width+ox)/float(count);
""") + """    }
}
W(i,value);
"""

# Conventional spelling aliases remain real native entry points.
for original in ["slice", "gather", "index_select", "repeat"]:
    bodies[original + "_backward"] = bodies[original + "_grad"]

for directory in ["CPU", "CUDA", "Compute", "Metal"]:
    (OPERATORS / directory).mkdir(parents=True, exist_ok=True)

cpu_registry = []
cuda_registry = []
for name, body in bodies.items():
    write(KERNELS / (name + ".inc"), "// " + name + " scalar implementation.\n" + body, encoding="utf-8")
    write(OPERATORS / "CPU" / (name + "_cpu.cpp"), f'''// Generated by Src/Kernels/generate.py; the scalar implementation is shared.
#include "../../Kernels/cpu_kernel.h"
namespace elman {{
void launch_{name}_cpu(uint32_t* out,const uint32_t* a,const uint32_t* b,const uint32_t* c,const KernelParams& p) {{
    for(int i=0;i<P(0);++i) {{
#include "../../Kernels/{name}.inc"
    }}
}}
}}
''', encoding="utf-8")
    if name not in {"cross_entropy", "cross_entropy_grad"}:
        write(OPERATORS / "CUDA" / (name + "_cuda.cu"), f'''// Generated by Src/Kernels/generate.py; the scalar implementation is shared.
#include "../../Backends/cuda_kernel.cuh"
namespace elman {{
__global__ void kernel_{name}_cuda(uint32_t* out,const uint32_t* a,const uint32_t* b,const uint32_t* c,KernelParams p) {{
    unsigned int element=blockIdx.x*blockDim.x+threadIdx.x;
    if(element<unsigned(P(0))) {{
    int i=int(element);
#include "../../Kernels/{name}.inc"
    }}
}}
void launch_{name}_cuda(uint32_t* out,const uint32_t* a,const uint32_t* b,const uint32_t* c,const KernelParams& p) {{
    if(P(0)>0)kernel_{name}_cuda<<<(unsigned(P(0))+255u)/256u,256>>>(out,a,b,c,p);
}}
}}
''', encoding="utf-8")
    write(OPERATORS / "Compute" / (name + "_compute.hlsl"), f'''// Generated by Src/Kernels/generate.py; the scalar implementation is shared.
#include "../../Backends/compute_kernel.hlsli"
[numthreads(64,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {{
    uint element=tid.x+tid.y*4194240u;
    if(element<uint(P(0))) {{
    int i=int(element);
#include "../../Kernels/{name}.inc"
    }}
}}
''', encoding="utf-8")
    write(OPERATORS / "Metal" / (name + "_metal.metal"), f'''// Generated by Src/Kernels/generate.py; the scalar implementation is shared.
#include "../../Backends/metal_kernel.h"
kernel void elman_main(uint element [[thread_position_in_grid]],device uint* out [[buffer(0)]],device const uint* a [[buffer(1)]],device const uint* b [[buffer(2)]],device const uint* c [[buffer(3)]],constant int* p [[buffer(4)]]) {{
    if(element<uint(P(0))) {{
    int i=int(element);
#include "../../Kernels/{name}.inc"
    }}
}}
''', encoding="utf-8")
    cpu_registry.append(f"ELMAN_CPU_KERNEL({name})\n")
    cuda_registry.append(f"ELMAN_CUDA_KERNEL({name})\n")
write(OPERATORS / "cpu_dispatch.inc", "".join(cpu_registry), encoding="utf-8")
write(OPERATORS / "cuda_dispatch.inc", "".join(cuda_registry), encoding="utf-8")
write(KERNELS / "operators.txt", "\n".join(bodies) + "\n", encoding="utf-8")
print(f"Generated {len(bodies)} operators for four native backends.")
