¬
Ù­
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
?
Mul
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
[
Split
	split_dim

value"T
output"T*	num_split"
	num_splitint(0"	
Ttype
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
°
TensorListFromTensor
tensor"element_dtype
element_shape"
shape_type/
output_handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListReserve
element_shape"
shape_type
num_elements(
handleéèelement_dtype"
element_dtypetype"

shape_typetype:
2	

TensorListStack
input_handle
element_shape
tensor"element_dtype"
element_dtypetype" 
num_elementsintÿÿÿÿÿÿÿÿÿ
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 

While

input2T
output2T"
T
list(type)("
condfunc"
bodyfunc" 
output_shapeslist(shape)
 "
parallel_iterationsint
"serve*2.9.22v2.9.1-132-g18960c44ad38Ô

Adam/lstm_4/lstm_cell_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_4/lstm_cell_4/bias/v

2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/v*
_output_shapes	
:*
dtype0
±
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
ª
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v*
_output_shapes
:	d*
dtype0

 Adam/lstm_4/lstm_cell_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/v

4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:d*
dtype0

Adam/lstm_4/lstm_cell_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/lstm_4/lstm_cell_4/bias/m

2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/lstm_4/lstm_cell_4/bias/m*
_output_shapes	
:*
dtype0
±
*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*;
shared_name,*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
ª
>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOpReadVariableOp*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m*
_output_shapes
:	d*
dtype0

 Adam/lstm_4/lstm_cell_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/lstm_4/lstm_cell_4/kernel/m

4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/lstm_4/lstm_cell_4/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:d*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	

lstm_4/lstm_cell_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namelstm_4/lstm_cell_4/bias

+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/bias*
_output_shapes	
:*
dtype0
£
#lstm_4/lstm_cell_4/recurrent_kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d*4
shared_name%#lstm_4/lstm_cell_4/recurrent_kernel

7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOpReadVariableOp#lstm_4/lstm_cell_4/recurrent_kernel*
_output_shapes
:	d*
dtype0

lstm_4/lstm_cell_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_namelstm_4/lstm_cell_4/kernel

-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOpReadVariableOplstm_4/lstm_cell_4/kernel*
_output_shapes
:	*
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:*
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:d*
dtype0

NoOpNoOp
ê-
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*¥-
value-B- B-
¤
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer
loss

signatures*
Á
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
'
0
1
 2
3
4*
'
0
1
 2
3
4*
* 
°
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
&trace_0
'trace_1
(trace_2
)trace_3* 
6
*trace_0
+trace_1
,trace_2
-trace_3* 
* 

.iter

/beta_1

0beta_2
	1decay
2learning_ratememfmgmh mivjvkvlvm vn*
* 

3serving_default* 

0
1
 2*

0
1
 2*
* 


4states
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
* 
ã
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H_random_generator
I
state_size

kernel
recurrent_kernel
 bias*
* 

0
1*

0
1*
* 

Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Otrace_0* 

Ptrace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUElstm_4/lstm_cell_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUE#lstm_4/lstm_cell_4/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUElstm_4/lstm_cell_4/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1*

Q0
R1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

0
1
 2*

0
1
 2*
* 

Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

Xtrace_0
Ytrace_1* 

Ztrace_0
[trace_1* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
\	variables
]	keras_api
	^total
	_count*
H
`	variables
a	keras_api
	btotal
	ccount
d
_fn_kwargs*
* 
* 
* 
* 
* 
* 
* 
* 
* 

^0
_1*

\	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

b0
c1*

`	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
{
VARIABLE_VALUEAdam/dense_4/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{
VARIABLE_VALUEAdam/dense_4/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUEAdam/dense_4/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|v
VARIABLE_VALUE Adam/lstm_4/lstm_cell_4/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
zt
VARIABLE_VALUEAdam/lstm_4/lstm_cell_4/bias/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_lstm_4_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
»
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_4_inputlstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/biasdense_4/kerneldense_4/bias*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *-
f(R&
$__inference_signature_wrapper_412399
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¦

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp-lstm_4/lstm_cell_4/kernel/Read/ReadVariableOp7lstm_4/lstm_cell_4/recurrent_kernel/Read/ReadVariableOp+lstm_4/lstm_cell_4/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/m/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/m/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp4Adam/lstm_4/lstm_cell_4/kernel/v/Read/ReadVariableOp>Adam/lstm_4/lstm_cell_4/recurrent_kernel/v/Read/ReadVariableOp2Adam/lstm_4/lstm_cell_4/bias/v/Read/ReadVariableOpConst*%
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *(
f#R!
__inference__traced_save_413567
Á
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biaslstm_4/lstm_cell_4/kernel#lstm_4/lstm_cell_4/recurrent_kernellstm_4/lstm_cell_4/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/dense_4/kernel/mAdam/dense_4/bias/m Adam/lstm_4/lstm_cell_4/kernel/m*Adam/lstm_4/lstm_cell_4/recurrent_kernel/mAdam/lstm_4/lstm_cell_4/bias/mAdam/dense_4/kernel/vAdam/dense_4/bias/v Adam/lstm_4/lstm_cell_4/kernel/v*Adam/lstm_4/lstm_cell_4/recurrent_kernel/vAdam/lstm_4/lstm_cell_4/bias/v*$
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__traced_restore_413649´à
íJ

B__inference_lstm_4_layer_call_and_return_conditional_losses_413210

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	?
,lstm_cell_4_matmul_1_readvariableop_resource:	d:
+lstm_cell_4_biasadd_readvariableop_resource:	
identity¢"lstm_cell_4/BiasAdd/ReadVariableOp¢!lstm_cell_4/MatMul/ReadVariableOp¢#lstm_cell_4/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitl
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_413125*
condR
while_cond_413124*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
K

B__inference_lstm_4_layer_call_and_return_conditional_losses_412920
inputs_0=
*lstm_cell_4_matmul_readvariableop_resource:	?
,lstm_cell_4_matmul_1_readvariableop_resource:	d:
+lstm_cell_4_biasadd_readvariableop_resource:	
identity¢"lstm_cell_4/BiasAdd/ReadVariableOp¢!lstm_cell_4/MatMul/ReadVariableOp¢#lstm_cell_4/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitl
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_412835*
condR
while_cond_412834*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
èZ
ý
H__inference_sequential_4_layer_call_and_return_conditional_losses_412731

inputsD
1lstm_4_lstm_cell_4_matmul_readvariableop_resource:	F
3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource:	dA
2lstm_4_lstm_cell_4_biasadd_readvariableop_resource:	8
&dense_4_matmul_readvariableop_resource:d5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp¢(lstm_4/lstm_cell_4/MatMul/ReadVariableOp¢*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp¢lstm_4/whileB
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_4/transpose	Transposeinputslstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0©
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:00lstm_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0£
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/zeros:output:02lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/MatMul:product:0%lstm_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
lstm_4/lstm_cell_4/BiasAddBiasAddlstm_4/lstm_cell_4/add:z:01lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0#lstm_4/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitz
lstm_4/lstm_cell_4/SigmoidSigmoid!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
lstm_4/lstm_cell_4/Sigmoid_1Sigmoid!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/mulMul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
lstm_4/lstm_cell_4/ReluRelu!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/mul_1Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/add_1AddV2lstm_4/lstm_cell_4/mul:z:0lstm_4/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
lstm_4/lstm_cell_4/Sigmoid_2Sigmoid!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/mul_2Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   e
#lstm_4/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0,lstm_4/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : á
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_4_lstm_cell_4_matmul_readvariableop_resource3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_412640*$
condR
lstm_4_while_cond_412639*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ë
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementso
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_4/MatMulMatMullstm_4/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*^lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp)^lstm_4/lstm_cell_4/MatMul/ReadVariableOp+^lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2V
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp2T
(lstm_4/lstm_cell_4/MatMul/ReadVariableOp(lstm_4/lstm_cell_4/MatMul/ReadVariableOp2X
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp2
lstm_4/whilelstm_4/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
µ
Ã
while_cond_412834
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_412834___redundant_placeholder04
0while_while_cond_412834___redundant_placeholder14
0while_while_cond_412834___redundant_placeholder24
0while_while_cond_412834___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Æ	
ô
C__inference_dense_4_layer_call_and_return_conditional_losses_412086

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
Æ	
ô
C__inference_dense_4_layer_call_and_return_conditional_losses_413374

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ð8

B__inference_lstm_4_layer_call_and_return_conditional_losses_411715

inputs%
lstm_cell_4_411631:	%
lstm_cell_4_411633:	d!
lstm_cell_4_411635:	
identity¢#lstm_cell_4/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskó
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_411631lstm_cell_4_411633lstm_cell_4_411635*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411630n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ´
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_411631lstm_cell_4_411633lstm_cell_4_411635*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_411645*
condR
while_cond_411644*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ô

G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411630

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
ÁP
Ç
%sequential_4_lstm_4_while_body_411472D
@sequential_4_lstm_4_while_sequential_4_lstm_4_while_loop_counterJ
Fsequential_4_lstm_4_while_sequential_4_lstm_4_while_maximum_iterations)
%sequential_4_lstm_4_while_placeholder+
'sequential_4_lstm_4_while_placeholder_1+
'sequential_4_lstm_4_while_placeholder_2+
'sequential_4_lstm_4_while_placeholder_3C
?sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1_0
{sequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor_0Y
Fsequential_4_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0:	[
Hsequential_4_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dV
Gsequential_4_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0:	&
"sequential_4_lstm_4_while_identity(
$sequential_4_lstm_4_while_identity_1(
$sequential_4_lstm_4_while_identity_2(
$sequential_4_lstm_4_while_identity_3(
$sequential_4_lstm_4_while_identity_4(
$sequential_4_lstm_4_while_identity_5A
=sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1}
ysequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensorW
Dsequential_4_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource:	Y
Fsequential_4_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource:	dT
Esequential_4_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:	¢<sequential_4/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp¢;sequential_4/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp¢=sequential_4/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp
Ksequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
=sequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem{sequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor_0%sequential_4_lstm_4_while_placeholderTsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0Ã
;sequential_4/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOpFsequential_4_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0ô
,sequential_4/lstm_4/while/lstm_cell_4/MatMulMatMulDsequential_4/lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:0Csequential_4/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÇ
=sequential_4/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOpHsequential_4_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0Û
.sequential_4/lstm_4/while/lstm_cell_4/MatMul_1MatMul'sequential_4_lstm_4_while_placeholder_2Esequential_4/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ×
)sequential_4/lstm_4/while/lstm_cell_4/addAddV26sequential_4/lstm_4/while/lstm_cell_4/MatMul:product:08sequential_4/lstm_4/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÁ
<sequential_4/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOpGsequential_4_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0à
-sequential_4/lstm_4/while/lstm_cell_4/BiasAddBiasAdd-sequential_4/lstm_4/while/lstm_cell_4/add:z:0Dsequential_4/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
5sequential_4/lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¨
+sequential_4/lstm_4/while/lstm_cell_4/splitSplit>sequential_4/lstm_4/while/lstm_cell_4/split/split_dim:output:06sequential_4/lstm_4/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split 
-sequential_4/lstm_4/while/lstm_cell_4/SigmoidSigmoid4sequential_4/lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
/sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid4sequential_4/lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÀ
)sequential_4/lstm_4/while/lstm_cell_4/mulMul3sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_1:y:0'sequential_4_lstm_4_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
*sequential_4/lstm_4/while/lstm_cell_4/ReluRelu4sequential_4/lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÑ
+sequential_4/lstm_4/while/lstm_cell_4/mul_1Mul1sequential_4/lstm_4/while/lstm_cell_4/Sigmoid:y:08sequential_4/lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÆ
+sequential_4/lstm_4/while/lstm_cell_4/add_1AddV2-sequential_4/lstm_4/while/lstm_cell_4/mul:z:0/sequential_4/lstm_4/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¢
/sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid4sequential_4/lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
,sequential_4/lstm_4/while/lstm_cell_4/Relu_1Relu/sequential_4/lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÕ
+sequential_4/lstm_4/while/lstm_cell_4/mul_2Mul3sequential_4/lstm_4/while/lstm_cell_4/Sigmoid_2:y:0:sequential_4/lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
Dsequential_4/lstm_4/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ¼
>sequential_4/lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItem'sequential_4_lstm_4_while_placeholder_1Msequential_4/lstm_4/while/TensorArrayV2Write/TensorListSetItem/index:output:0/sequential_4/lstm_4/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒa
sequential_4/lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :
sequential_4/lstm_4/while/addAddV2%sequential_4_lstm_4_while_placeholder(sequential_4/lstm_4/while/add/y:output:0*
T0*
_output_shapes
: c
!sequential_4/lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :·
sequential_4/lstm_4/while/add_1AddV2@sequential_4_lstm_4_while_sequential_4_lstm_4_while_loop_counter*sequential_4/lstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: 
"sequential_4/lstm_4/while/IdentityIdentity#sequential_4/lstm_4/while/add_1:z:0^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: º
$sequential_4/lstm_4/while/Identity_1IdentityFsequential_4_lstm_4_while_sequential_4_lstm_4_while_maximum_iterations^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: 
$sequential_4/lstm_4/while/Identity_2Identity!sequential_4/lstm_4/while/add:z:0^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: Â
$sequential_4/lstm_4/while/Identity_3IdentityNsequential_4/lstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^sequential_4/lstm_4/while/NoOp*
T0*
_output_shapes
: ´
$sequential_4/lstm_4/while/Identity_4Identity/sequential_4/lstm_4/while/lstm_cell_4/mul_2:z:0^sequential_4/lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
$sequential_4/lstm_4/while/Identity_5Identity/sequential_4/lstm_4/while/lstm_cell_4/add_1:z:0^sequential_4/lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
sequential_4/lstm_4/while/NoOpNoOp=^sequential_4/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp<^sequential_4/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp>^sequential_4/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "Q
"sequential_4_lstm_4_while_identity+sequential_4/lstm_4/while/Identity:output:0"U
$sequential_4_lstm_4_while_identity_1-sequential_4/lstm_4/while/Identity_1:output:0"U
$sequential_4_lstm_4_while_identity_2-sequential_4/lstm_4/while/Identity_2:output:0"U
$sequential_4_lstm_4_while_identity_3-sequential_4/lstm_4/while/Identity_3:output:0"U
$sequential_4_lstm_4_while_identity_4-sequential_4/lstm_4/while/Identity_4:output:0"U
$sequential_4_lstm_4_while_identity_5-sequential_4/lstm_4/while/Identity_5:output:0"
Esequential_4_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resourceGsequential_4_lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0"
Fsequential_4_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resourceHsequential_4_lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0"
Dsequential_4_lstm_4_while_lstm_cell_4_matmul_readvariableop_resourceFsequential_4_lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0"
=sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1?sequential_4_lstm_4_while_sequential_4_lstm_4_strided_slice_1_0"ø
ysequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor{sequential_4_lstm_4_while_tensorarrayv2read_tensorlistgetitem_sequential_4_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2|
<sequential_4/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp<sequential_4/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp2z
;sequential_4/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp;sequential_4/lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp2~
=sequential_4/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp=sequential_4/lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ì
õ
,__inference_lstm_cell_4_layer_call_fn_413408

inputs
states_0
states_1
unknown:	
	unknown_0:	d
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411778o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1

¶
'__inference_lstm_4_layer_call_fn_412753
inputs_0
unknown:	
	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_411908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
µ
Ã
while_cond_412979
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_412979___redundant_placeholder04
0while_while_cond_412979___redundant_placeholder14
0while_while_cond_412979___redundant_placeholder24
0while_while_cond_412979___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ô

G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411778

inputs

states
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0n
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_namestates
Ü

G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413472

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
æ
Ó
%sequential_4_lstm_4_while_cond_411471D
@sequential_4_lstm_4_while_sequential_4_lstm_4_while_loop_counterJ
Fsequential_4_lstm_4_while_sequential_4_lstm_4_while_maximum_iterations)
%sequential_4_lstm_4_while_placeholder+
'sequential_4_lstm_4_while_placeholder_1+
'sequential_4_lstm_4_while_placeholder_2+
'sequential_4_lstm_4_while_placeholder_3F
Bsequential_4_lstm_4_while_less_sequential_4_lstm_4_strided_slice_1\
Xsequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_411471___redundant_placeholder0\
Xsequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_411471___redundant_placeholder1\
Xsequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_411471___redundant_placeholder2\
Xsequential_4_lstm_4_while_sequential_4_lstm_4_while_cond_411471___redundant_placeholder3&
"sequential_4_lstm_4_while_identity
²
sequential_4/lstm_4/while/LessLess%sequential_4_lstm_4_while_placeholderBsequential_4_lstm_4_while_less_sequential_4_lstm_4_strided_slice_1*
T0*
_output_shapes
: s
"sequential_4/lstm_4/while/IdentityIdentity"sequential_4/lstm_4/while/Less:z:0*
T0
*
_output_shapes
: "Q
"sequential_4_lstm_4_while_identity+sequential_4/lstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_412188
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_412188___redundant_placeholder04
0while_while_cond_412188___redundant_placeholder14
0while_while_cond_412188___redundant_placeholder24
0while_while_cond_412188___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ø
´
'__inference_lstm_4_layer_call_fn_412775

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_412274o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ø
ð
-__inference_sequential_4_layer_call_fn_412429

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_412316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä8
Ç
while_body_412835
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	G
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	E
2while_lstm_cell_4_matmul_1_readvariableop_resource:	d@
1while_lstm_cell_4_biasadd_readvariableop_resource:	¢(while/lstm_cell_4/BiasAdd/ReadVariableOp¢'while/lstm_cell_4/MatMul/ReadVariableOp¢)while/lstm_cell_4/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¤
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitx
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Ûa

"__inference__traced_restore_413649
file_prefix1
assignvariableop_dense_4_kernel:d-
assignvariableop_1_dense_4_bias:?
,assignvariableop_2_lstm_4_lstm_cell_4_kernel:	I
6assignvariableop_3_lstm_4_lstm_cell_4_recurrent_kernel:	d9
*assignvariableop_4_lstm_4_lstm_cell_4_bias:	&
assignvariableop_5_adam_iter:	 (
assignvariableop_6_adam_beta_1: (
assignvariableop_7_adam_beta_2: '
assignvariableop_8_adam_decay: /
%assignvariableop_9_adam_learning_rate: %
assignvariableop_10_total_1: %
assignvariableop_11_count_1: #
assignvariableop_12_total: #
assignvariableop_13_count: ;
)assignvariableop_14_adam_dense_4_kernel_m:d5
'assignvariableop_15_adam_dense_4_bias_m:G
4assignvariableop_16_adam_lstm_4_lstm_cell_4_kernel_m:	Q
>assignvariableop_17_adam_lstm_4_lstm_cell_4_recurrent_kernel_m:	dA
2assignvariableop_18_adam_lstm_4_lstm_cell_4_bias_m:	;
)assignvariableop_19_adam_dense_4_kernel_v:d5
'assignvariableop_20_adam_dense_4_bias_v:G
4assignvariableop_21_adam_lstm_4_lstm_cell_4_kernel_v:	Q
>assignvariableop_22_adam_lstm_4_lstm_cell_4_recurrent_kernel_v:	dA
2assignvariableop_23_adam_lstm_4_lstm_cell_4_bias_v:	
identity_25¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*º
value°B­B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¢
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*x
_output_shapesf
d:::::::::::::::::::::::::*'
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp,assignvariableop_2_lstm_4_lstm_cell_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_3AssignVariableOp6assignvariableop_3_lstm_4_lstm_cell_4_recurrent_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp*assignvariableop_4_lstm_4_lstm_cell_4_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_5AssignVariableOpassignvariableop_5_adam_iterIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_beta_1Identity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_2Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_decayIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp%assignvariableop_9_adam_learning_rateIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOpassignvariableop_10_total_1Identity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_count_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_totalIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_countIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp)assignvariableop_14_adam_dense_4_kernel_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp'assignvariableop_15_adam_dense_4_bias_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_16AssignVariableOp4assignvariableop_16_adam_lstm_4_lstm_cell_4_kernel_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_17AssignVariableOp>assignvariableop_17_adam_lstm_4_lstm_cell_4_recurrent_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_18AssignVariableOp2assignvariableop_18_adam_lstm_4_lstm_cell_4_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp)assignvariableop_19_adam_dense_4_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_4_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_21AssignVariableOp4assignvariableop_21_adam_lstm_4_lstm_cell_4_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¯
AssignVariableOp_22AssignVariableOp>assignvariableop_22_adam_lstm_4_lstm_cell_4_recurrent_kernel_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_23AssignVariableOp2assignvariableop_23_adam_lstm_4_lstm_cell_4_bias_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ß
Identity_24Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_25IdentityIdentity_24:output:0^NoOp_1*
T0*
_output_shapes
: Ì
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_25Identity_25:output:0*E
_input_shapes4
2: : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
À
Ì
H__inference_sequential_4_layer_call_and_return_conditional_losses_412093

inputs 
lstm_4_412069:	 
lstm_4_412071:	d
lstm_4_412073:	 
dense_4_412087:d
dense_4_412089:
identity¢dense_4/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCallü
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_412069lstm_4_412071lstm_4_412073*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_412068
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_412087dense_4_412089*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_412086w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_4/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä8
Ç
while_body_413270
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	G
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	E
2while_lstm_cell_4_matmul_1_readvariableop_resource:	d@
1while_lstm_cell_4_biasadd_readvariableop_resource:	¢(while/lstm_cell_4/BiasAdd/ReadVariableOp¢'while/lstm_cell_4/MatMul/ReadVariableOp¢)while/lstm_cell_4/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¤
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitx
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
A
§

lstm_4_while_body_412489*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0:	N
;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dI
:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0:	
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorJ
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource:	L
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource:	dG
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:	¢/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp¢.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp¢0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0©
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Í
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0´
!lstm_4/while/lstm_cell_4/MatMul_1MatMullstm_4_while_placeholder_28lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/MatMul:product:0+lstm_4/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¹
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd lstm_4/while/lstm_cell_4/add:z:07lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:0)lstm_4/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
 lstm_4/while/lstm_cell_4/SigmoidSigmoid'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/lstm_cell_4/mulMul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/lstm_cell_4/ReluRelu'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
lstm_4/while/lstm_cell_4/mul_1Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/lstm_cell_4/add_1AddV2 lstm_4/while/lstm_cell_4/mul:z:0"lstm_4/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd®
lstm_4/while/lstm_cell_4/mul_2Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
7lstm_4/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1@lstm_4/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_4/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_2:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_1:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdé
lstm_4/while/NoOpNoOp0^lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp/^lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp1^lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"v
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0"x
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0"t
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0"Ä
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2b
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp2`
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp2d
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
A
§

lstm_4_while_body_412640*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3)
%lstm_4_while_lstm_4_strided_slice_1_0e
alstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0L
9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0:	N
;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dI
:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0:	
lstm_4_while_identity
lstm_4_while_identity_1
lstm_4_while_identity_2
lstm_4_while_identity_3
lstm_4_while_identity_4
lstm_4_while_identity_5'
#lstm_4_while_lstm_4_strided_slice_1c
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensorJ
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource:	L
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource:	dG
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:	¢/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp¢.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp¢0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp
>lstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   É
0lstm_4/while/TensorArrayV2Read/TensorListGetItemTensorListGetItemalstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0lstm_4_while_placeholderGlstm_4/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0©
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0Í
lstm_4/while/lstm_cell_4/MatMulMatMul7lstm_4/while/TensorArrayV2Read/TensorListGetItem:item:06lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ­
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0´
!lstm_4/while/lstm_cell_4/MatMul_1MatMullstm_4_while_placeholder_28lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ°
lstm_4/while/lstm_cell_4/addAddV2)lstm_4/while/lstm_cell_4/MatMul:product:0+lstm_4/while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ§
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¹
 lstm_4/while/lstm_cell_4/BiasAddBiasAdd lstm_4/while/lstm_cell_4/add:z:07lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
(lstm_4/while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/lstm_cell_4/splitSplit1lstm_4/while/lstm_cell_4/split/split_dim:output:0)lstm_4/while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
 lstm_4/while/lstm_cell_4/SigmoidSigmoid'lstm_4/while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/while/lstm_cell_4/Sigmoid_1Sigmoid'lstm_4/while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/lstm_cell_4/mulMul&lstm_4/while/lstm_cell_4/Sigmoid_1:y:0lstm_4_while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/lstm_cell_4/ReluRelu'lstm_4/while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdª
lstm_4/while/lstm_cell_4/mul_1Mul$lstm_4/while/lstm_cell_4/Sigmoid:y:0+lstm_4/while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/lstm_cell_4/add_1AddV2 lstm_4/while/lstm_cell_4/mul:z:0"lstm_4/while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/while/lstm_cell_4/Sigmoid_2Sigmoid'lstm_4/while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd}
lstm_4/while/lstm_cell_4/Relu_1Relu"lstm_4/while/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd®
lstm_4/while/lstm_cell_4/mul_2Mul&lstm_4/while/lstm_cell_4/Sigmoid_2:y:0-lstm_4/while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdy
7lstm_4/while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
1lstm_4/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemlstm_4_while_placeholder_1@lstm_4/while/TensorArrayV2Write/TensorListSetItem/index:output:0"lstm_4/while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒT
lstm_4/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :q
lstm_4/while/addAddV2lstm_4_while_placeholderlstm_4/while/add/y:output:0*
T0*
_output_shapes
: V
lstm_4/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :
lstm_4/while/add_1AddV2&lstm_4_while_lstm_4_while_loop_counterlstm_4/while/add_1/y:output:0*
T0*
_output_shapes
: n
lstm_4/while/IdentityIdentitylstm_4/while/add_1:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_1Identity,lstm_4_while_lstm_4_while_maximum_iterations^lstm_4/while/NoOp*
T0*
_output_shapes
: n
lstm_4/while/Identity_2Identitylstm_4/while/add:z:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_3IdentityAlstm_4/while/TensorArrayV2Write/TensorListSetItem:output_handle:0^lstm_4/while/NoOp*
T0*
_output_shapes
: 
lstm_4/while/Identity_4Identity"lstm_4/while/lstm_cell_4/mul_2:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/while/Identity_5Identity"lstm_4/while/lstm_cell_4/add_1:z:0^lstm_4/while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdé
lstm_4/while/NoOpNoOp0^lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp/^lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp1^lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "7
lstm_4_while_identitylstm_4/while/Identity:output:0";
lstm_4_while_identity_1 lstm_4/while/Identity_1:output:0";
lstm_4_while_identity_2 lstm_4/while/Identity_2:output:0";
lstm_4_while_identity_3 lstm_4/while/Identity_3:output:0";
lstm_4_while_identity_4 lstm_4/while/Identity_4:output:0";
lstm_4_while_identity_5 lstm_4/while/Identity_5:output:0"L
#lstm_4_while_lstm_4_strided_slice_1%lstm_4_while_lstm_4_strided_slice_1_0"v
8lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource:lstm_4_while_lstm_cell_4_biasadd_readvariableop_resource_0"x
9lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource;lstm_4_while_lstm_cell_4_matmul_1_readvariableop_resource_0"t
7lstm_4_while_lstm_cell_4_matmul_readvariableop_resource9lstm_4_while_lstm_cell_4_matmul_readvariableop_resource_0"Ä
_lstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensoralstm_4_while_tensorarrayv2read_tensorlistgetitem_lstm_4_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2b
/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp/lstm_4/while/lstm_cell_4/BiasAdd/ReadVariableOp2`
.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp.lstm_4/while/lstm_cell_4/MatMul/ReadVariableOp2d
0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp0lstm_4/while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ì
õ
,__inference_lstm_cell_4_layer_call_fn_413391

inputs
states_0
states_1
unknown:	
	unknown_0:	d
	unknown_1:	
identity

identity_1

identity_2¢StatefulPartitionedCallª
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1unknown	unknown_0	unknown_1*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411630o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
K

B__inference_lstm_4_layer_call_and_return_conditional_losses_413065
inputs_0=
*lstm_cell_4_matmul_readvariableop_resource:	?
,lstm_cell_4_matmul_1_readvariableop_resource:	d:
+lstm_cell_4_biasadd_readvariableop_resource:	
identity¢"lstm_cell_4/BiasAdd/ReadVariableOp¢!lstm_cell_4/MatMul/ReadVariableOp¢#lstm_cell_4/MatMul_1/ReadVariableOp¢while=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitl
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_412980*
condR
while_cond_412979*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0
ð8

B__inference_lstm_4_layer_call_and_return_conditional_losses_411908

inputs%
lstm_cell_4_411824:	%
lstm_cell_4_411826:	d!
lstm_cell_4_411828:	
identity¢#lstm_cell_4/StatefulPartitionedCall¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskó
#lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0lstm_cell_4_411824lstm_cell_4_411826lstm_cell_4_411828*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411778n
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ´
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0lstm_cell_4_411824lstm_cell_4_411826lstm_cell_4_411828*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_411838*
condR
while_cond_411837*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
NoOpNoOp$^lstm_cell_4/StatefulPartitionedCall^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 2J
#lstm_cell_4/StatefulPartitionedCall#lstm_cell_4/StatefulPartitionedCall2
whilewhile:\ X
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä8
Ç
while_body_411983
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	G
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	E
2while_lstm_cell_4_matmul_1_readvariableop_resource:	d@
1while_lstm_cell_4_biasadd_readvariableop_resource:	¢(while/lstm_cell_4/BiasAdd/ReadVariableOp¢'while/lstm_cell_4/MatMul/ReadVariableOp¢)while/lstm_cell_4/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¤
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitx
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_413124
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_413124___redundant_placeholder04
0while_while_cond_413124___redundant_placeholder14
0while_while_cond_413124___redundant_placeholder24
0while_while_cond_413124___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ã

(__inference_dense_4_layer_call_fn_413364

inputs
unknown:d
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_412086o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿd: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
 
_user_specified_nameinputs
ø
´
'__inference_lstm_4_layer_call_fn_412764

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_412068o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
º
í
$__inference_signature_wrapper_412399
lstm_4_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d
	unknown_3:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 **
f%R#
!__inference__wrapped_model_411563o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_4_input
µ
Ã
while_cond_413269
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_413269___redundant_placeholder04
0while_while_cond_413269___redundant_placeholder14
0while_while_cond_413269___redundant_placeholder24
0while_while_cond_413269___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
Ò
Ò
H__inference_sequential_4_layer_call_and_return_conditional_losses_412360
lstm_4_input 
lstm_4_412347:	 
lstm_4_412349:	d
lstm_4_412351:	 
dense_4_412354:d
dense_4_412356:
identity¢dense_4/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCall
lstm_4/StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputlstm_4_412347lstm_4_412349lstm_4_412351*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_412068
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_412354dense_4_412356*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_412086w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_4/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_4_input
Ø
ð
-__inference_sequential_4_layer_call_fn_412414

inputs
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_412093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü

G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413440

inputs
states_0
states_11
matmul_readvariableop_resource:	3
 matmul_1_readvariableop_resource:	d.
biasadd_readvariableop_resource:	
identity

identity_1

identity_2¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOp¢MatMul_1/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿy
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0p
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0n
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :¶
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitT
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdU
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdN
ReluRelusplit:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdT
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdV
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdK
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdX
IdentityIdentity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_1Identity	mul_2:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdZ

Identity_2Identity	add_1:z:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"
_user_specified_name
states/1
Ýl
ë
!__inference__wrapped_model_411563
lstm_4_inputQ
>sequential_4_lstm_4_lstm_cell_4_matmul_readvariableop_resource:	S
@sequential_4_lstm_4_lstm_cell_4_matmul_1_readvariableop_resource:	dN
?sequential_4_lstm_4_lstm_cell_4_biasadd_readvariableop_resource:	E
3sequential_4_dense_4_matmul_readvariableop_resource:dB
4sequential_4_dense_4_biasadd_readvariableop_resource:
identity¢+sequential_4/dense_4/BiasAdd/ReadVariableOp¢*sequential_4/dense_4/MatMul/ReadVariableOp¢6sequential_4/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp¢5sequential_4/lstm_4/lstm_cell_4/MatMul/ReadVariableOp¢7sequential_4/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp¢sequential_4/lstm_4/whileU
sequential_4/lstm_4/ShapeShapelstm_4_input*
T0*
_output_shapes
:q
'sequential_4/lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: s
)sequential_4/lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:s
)sequential_4/lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
!sequential_4/lstm_4/strided_sliceStridedSlice"sequential_4/lstm_4/Shape:output:00sequential_4/lstm_4/strided_slice/stack:output:02sequential_4/lstm_4/strided_slice/stack_1:output:02sequential_4/lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskd
"sequential_4/lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d¯
 sequential_4/lstm_4/zeros/packedPack*sequential_4/lstm_4/strided_slice:output:0+sequential_4/lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:d
sequential_4/lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ¨
sequential_4/lstm_4/zerosFill)sequential_4/lstm_4/zeros/packed:output:0(sequential_4/lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
$sequential_4/lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d³
"sequential_4/lstm_4/zeros_1/packedPack*sequential_4/lstm_4/strided_slice:output:0-sequential_4/lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:f
!sequential_4/lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    ®
sequential_4/lstm_4/zeros_1Fill+sequential_4/lstm_4/zeros_1/packed:output:0*sequential_4/lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdw
"sequential_4/lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          
sequential_4/lstm_4/transpose	Transposelstm_4_input+sequential_4/lstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿl
sequential_4/lstm_4/Shape_1Shape!sequential_4/lstm_4/transpose:y:0*
T0*
_output_shapes
:s
)sequential_4/lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
#sequential_4/lstm_4/strided_slice_1StridedSlice$sequential_4/lstm_4/Shape_1:output:02sequential_4/lstm_4/strided_slice_1/stack:output:04sequential_4/lstm_4/strided_slice_1/stack_1:output:04sequential_4/lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskz
/sequential_4/lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿð
!sequential_4/lstm_4/TensorArrayV2TensorListReserve8sequential_4/lstm_4/TensorArrayV2/element_shape:output:0,sequential_4/lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
Isequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
;sequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensor!sequential_4/lstm_4/transpose:y:0Rsequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒs
)sequential_4/lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:u
+sequential_4/lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Í
#sequential_4/lstm_4/strided_slice_2StridedSlice!sequential_4/lstm_4/transpose:y:02sequential_4/lstm_4/strided_slice_2/stack:output:04sequential_4/lstm_4/strided_slice_2/stack_1:output:04sequential_4/lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_maskµ
5sequential_4/lstm_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp>sequential_4_lstm_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0Ð
&sequential_4/lstm_4/lstm_cell_4/MatMulMatMul,sequential_4/lstm_4/strided_slice_2:output:0=sequential_4/lstm_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¹
7sequential_4/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp@sequential_4_lstm_4_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0Ê
(sequential_4/lstm_4/lstm_cell_4/MatMul_1MatMul"sequential_4/lstm_4/zeros:output:0?sequential_4/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÅ
#sequential_4/lstm_4/lstm_cell_4/addAddV20sequential_4/lstm_4/lstm_cell_4/MatMul:product:02sequential_4/lstm_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ³
6sequential_4/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp?sequential_4_lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Î
'sequential_4/lstm_4/lstm_cell_4/BiasAddBiasAdd'sequential_4/lstm_4/lstm_cell_4/add:z:0>sequential_4/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿq
/sequential_4/lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :
%sequential_4/lstm_4/lstm_cell_4/splitSplit8sequential_4/lstm_4/lstm_cell_4/split/split_dim:output:00sequential_4/lstm_4/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_split
'sequential_4/lstm_4/lstm_cell_4/SigmoidSigmoid.sequential_4/lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)sequential_4/lstm_4/lstm_cell_4/Sigmoid_1Sigmoid.sequential_4/lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd±
#sequential_4/lstm_4/lstm_cell_4/mulMul-sequential_4/lstm_4/lstm_cell_4/Sigmoid_1:y:0$sequential_4/lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
$sequential_4/lstm_4/lstm_cell_4/ReluRelu.sequential_4/lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd¿
%sequential_4/lstm_4/lstm_cell_4/mul_1Mul+sequential_4/lstm_4/lstm_cell_4/Sigmoid:y:02sequential_4/lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd´
%sequential_4/lstm_4/lstm_cell_4/add_1AddV2'sequential_4/lstm_4/lstm_cell_4/mul:z:0)sequential_4/lstm_4/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
)sequential_4/lstm_4/lstm_cell_4/Sigmoid_2Sigmoid.sequential_4/lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
&sequential_4/lstm_4/lstm_cell_4/Relu_1Relu)sequential_4/lstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÃ
%sequential_4/lstm_4/lstm_cell_4/mul_2Mul-sequential_4/lstm_4/lstm_cell_4/Sigmoid_2:y:04sequential_4/lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
1sequential_4/lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   r
0sequential_4/lstm_4/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :
#sequential_4/lstm_4/TensorArrayV2_1TensorListReserve:sequential_4/lstm_4/TensorArrayV2_1/element_shape:output:09sequential_4/lstm_4/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒZ
sequential_4/lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : w
,sequential_4/lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿh
&sequential_4/lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 
sequential_4/lstm_4/whileWhile/sequential_4/lstm_4/while/loop_counter:output:05sequential_4/lstm_4/while/maximum_iterations:output:0!sequential_4/lstm_4/time:output:0,sequential_4/lstm_4/TensorArrayV2_1:handle:0"sequential_4/lstm_4/zeros:output:0$sequential_4/lstm_4/zeros_1:output:0,sequential_4/lstm_4/strided_slice_1:output:0Ksequential_4/lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:0>sequential_4_lstm_4_lstm_cell_4_matmul_readvariableop_resource@sequential_4_lstm_4_lstm_cell_4_matmul_1_readvariableop_resource?sequential_4_lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *1
body)R'
%sequential_4_lstm_4_while_body_411472*1
cond)R'
%sequential_4_lstm_4_while_cond_411471*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
Dsequential_4/lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   
6sequential_4/lstm_4/TensorArrayV2Stack/TensorListStackTensorListStack"sequential_4/lstm_4/while:output:3Msequential_4/lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elements|
)sequential_4/lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿu
+sequential_4/lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+sequential_4/lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ë
#sequential_4/lstm_4/strided_slice_3StridedSlice?sequential_4/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:02sequential_4/lstm_4/strided_slice_3/stack:output:04sequential_4/lstm_4/strided_slice_3/stack_1:output:04sequential_4/lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_masky
$sequential_4/lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          Ò
sequential_4/lstm_4/transpose_1	Transpose?sequential_4/lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0-sequential_4/lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
sequential_4/lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
*sequential_4/dense_4/MatMul/ReadVariableOpReadVariableOp3sequential_4_dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0¹
sequential_4/dense_4/MatMulMatMul,sequential_4/lstm_4/strided_slice_3:output:02sequential_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
+sequential_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp4sequential_4_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0µ
sequential_4/dense_4/BiasAddBiasAdd%sequential_4/dense_4/MatMul:product:03sequential_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿt
IdentityIdentity%sequential_4/dense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿè
NoOpNoOp,^sequential_4/dense_4/BiasAdd/ReadVariableOp+^sequential_4/dense_4/MatMul/ReadVariableOp7^sequential_4/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp6^sequential_4/lstm_4/lstm_cell_4/MatMul/ReadVariableOp8^sequential_4/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp^sequential_4/lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2Z
+sequential_4/dense_4/BiasAdd/ReadVariableOp+sequential_4/dense_4/BiasAdd/ReadVariableOp2X
*sequential_4/dense_4/MatMul/ReadVariableOp*sequential_4/dense_4/MatMul/ReadVariableOp2p
6sequential_4/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp6sequential_4/lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp2n
5sequential_4/lstm_4/lstm_cell_4/MatMul/ReadVariableOp5sequential_4/lstm_4/lstm_cell_4/MatMul/ReadVariableOp2r
7sequential_4/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp7sequential_4/lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp26
sequential_4/lstm_4/whilesequential_4/lstm_4/while:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_4_input
ê
ö
-__inference_sequential_4_layer_call_fn_412344
lstm_4_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_412316o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_4_input
Ç7
ä

__inference__traced_save_413567
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop8
4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableopB
>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop6
2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop?
;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopI
Esavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop=
9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*º
value°B­B6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*E
value<B:B B B B B B B B B B B B B B B B B B B B B B B B B à

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop4savev2_lstm_4_lstm_cell_4_kernel_read_readvariableop>savev2_lstm_4_lstm_cell_4_recurrent_kernel_read_readvariableop2savev2_lstm_4_lstm_cell_4_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_m_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_m_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop;savev2_adam_lstm_4_lstm_cell_4_kernel_v_read_readvariableopEsavev2_adam_lstm_4_lstm_cell_4_recurrent_kernel_v_read_readvariableop9savev2_adam_lstm_4_lstm_cell_4_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *'
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*²
_input_shapes 
: :d::	:	d:: : : : : : : : : :d::	:	d::d::	:	d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	d:!

_output_shapes	
::$ 

_output_shapes

:d: 

_output_shapes
::%!

_output_shapes
:	:%!

_output_shapes
:	d:!

_output_shapes	
::

_output_shapes
: 
íJ

B__inference_lstm_4_layer_call_and_return_conditional_losses_412274

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	?
,lstm_cell_4_matmul_1_readvariableop_resource:	d:
+lstm_cell_4_biasadd_readvariableop_resource:	
identity¢"lstm_cell_4/BiasAdd/ReadVariableOp¢!lstm_cell_4/MatMul/ReadVariableOp¢#lstm_cell_4/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitl
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_412189*
condR
while_cond_412188*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ä8
Ç
while_body_412980
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	G
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	E
2while_lstm_cell_4_matmul_1_readvariableop_resource:	d@
1while_lstm_cell_4_biasadd_readvariableop_resource:	¢(while/lstm_cell_4/BiasAdd/ReadVariableOp¢'while/lstm_cell_4/MatMul/ReadVariableOp¢)while/lstm_cell_4/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¤
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitx
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ä8
Ç
while_body_412189
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	G
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	E
2while_lstm_cell_4_matmul_1_readvariableop_resource:	d@
1while_lstm_cell_4_biasadd_readvariableop_resource:	¢(while/lstm_cell_4/BiasAdd/ReadVariableOp¢'while/lstm_cell_4/MatMul/ReadVariableOp¢)while/lstm_cell_4/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¤
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitx
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
èZ
ý
H__inference_sequential_4_layer_call_and_return_conditional_losses_412580

inputsD
1lstm_4_lstm_cell_4_matmul_readvariableop_resource:	F
3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource:	dA
2lstm_4_lstm_cell_4_biasadd_readvariableop_resource:	8
&dense_4_matmul_readvariableop_resource:d5
'dense_4_biasadd_readvariableop_resource:
identity¢dense_4/BiasAdd/ReadVariableOp¢dense_4/MatMul/ReadVariableOp¢)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp¢(lstm_4/lstm_cell_4/MatMul/ReadVariableOp¢*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp¢lstm_4/whileB
lstm_4/ShapeShapeinputs*
T0*
_output_shapes
:d
lstm_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: f
lstm_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:f
lstm_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ô
lstm_4/strided_sliceStridedSlicelstm_4/Shape:output:0#lstm_4/strided_slice/stack:output:0%lstm_4/strided_slice/stack_1:output:0%lstm_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskW
lstm_4/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_4/zeros/packedPacklstm_4/strided_slice:output:0lstm_4/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:W
lstm_4/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zerosFilllstm_4/zeros/packed:output:0lstm_4/zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdY
lstm_4/zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :d
lstm_4/zeros_1/packedPacklstm_4/strided_slice:output:0 lstm_4/zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:Y
lstm_4/zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    
lstm_4/zeros_1Filllstm_4/zeros_1/packed:output:0lstm_4/zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdj
lstm_4/transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          {
lstm_4/transpose	Transposeinputslstm_4/transpose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿR
lstm_4/Shape_1Shapelstm_4/transpose:y:0*
T0*
_output_shapes
:f
lstm_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:þ
lstm_4/strided_slice_1StridedSlicelstm_4/Shape_1:output:0%lstm_4/strided_slice_1/stack:output:0'lstm_4/strided_slice_1/stack_1:output:0'lstm_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskm
"lstm_4/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿÉ
lstm_4/TensorArrayV2TensorListReserve+lstm_4/TensorArrayV2/element_shape:output:0lstm_4/strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
<lstm_4/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   õ
.lstm_4/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_4/transpose:y:0Elstm_4/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒf
lstm_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:h
lstm_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
lstm_4/strided_slice_2StridedSlicelstm_4/transpose:y:0%lstm_4/strided_slice_2/stack:output:0'lstm_4/strided_slice_2/stack_1:output:0'lstm_4/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
(lstm_4/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp1lstm_4_lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0©
lstm_4/lstm_cell_4/MatMulMatMullstm_4/strided_slice_2:output:00lstm_4/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0£
lstm_4/lstm_cell_4/MatMul_1MatMullstm_4/zeros:output:02lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_4/lstm_cell_4/addAddV2#lstm_4/lstm_cell_4/MatMul:product:0%lstm_4/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0§
lstm_4/lstm_cell_4/BiasAddBiasAddlstm_4/lstm_cell_4/add:z:01lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
"lstm_4/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ï
lstm_4/lstm_cell_4/splitSplit+lstm_4/lstm_cell_4/split/split_dim:output:0#lstm_4/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitz
lstm_4/lstm_cell_4/SigmoidSigmoid!lstm_4/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
lstm_4/lstm_cell_4/Sigmoid_1Sigmoid!lstm_4/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/mulMul lstm_4/lstm_cell_4/Sigmoid_1:y:0lstm_4/zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdt
lstm_4/lstm_cell_4/ReluRelu!lstm_4/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/mul_1Mullstm_4/lstm_cell_4/Sigmoid:y:0%lstm_4/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/add_1AddV2lstm_4/lstm_cell_4/mul:z:0lstm_4/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd|
lstm_4/lstm_cell_4/Sigmoid_2Sigmoid!lstm_4/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdq
lstm_4/lstm_cell_4/Relu_1Relulstm_4/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_4/lstm_cell_4/mul_2Mul lstm_4/lstm_cell_4/Sigmoid_2:y:0'lstm_4/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
$lstm_4/TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   e
#lstm_4/TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_4/TensorArrayV2_1TensorListReserve-lstm_4/TensorArrayV2_1/element_shape:output:0,lstm_4/TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒM
lstm_4/timeConst*
_output_shapes
: *
dtype0*
value	B : j
lstm_4/while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ[
lstm_4/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : á
lstm_4/whileWhile"lstm_4/while/loop_counter:output:0(lstm_4/while/maximum_iterations:output:0lstm_4/time:output:0lstm_4/TensorArrayV2_1:handle:0lstm_4/zeros:output:0lstm_4/zeros_1:output:0lstm_4/strided_slice_1:output:0>lstm_4/TensorArrayUnstack/TensorListFromTensor:output_handle:01lstm_4_lstm_cell_4_matmul_readvariableop_resource3lstm_4_lstm_cell_4_matmul_1_readvariableop_resource2lstm_4_lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *$
bodyR
lstm_4_while_body_412489*$
condR
lstm_4_while_cond_412488*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
7lstm_4/TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ë
)lstm_4/TensorArrayV2Stack/TensorListStackTensorListStacklstm_4/while:output:3@lstm_4/TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementso
lstm_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿh
lstm_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: h
lstm_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ª
lstm_4/strided_slice_3StridedSlice2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_4/strided_slice_3/stack:output:0'lstm_4/strided_slice_3/stack_1:output:0'lstm_4/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maskl
lstm_4/transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          «
lstm_4/transpose_1	Transpose2lstm_4/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_4/transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿdb
lstm_4/runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0
dense_4/MatMulMatMullstm_4/strided_slice_3:output:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
IdentityIdentitydense_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*^lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp)^lstm_4/lstm_cell_4/MatMul/ReadVariableOp+^lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp^lstm_4/while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2V
)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp)lstm_4/lstm_cell_4/BiasAdd/ReadVariableOp2T
(lstm_4/lstm_cell_4/MatMul/ReadVariableOp(lstm_4/lstm_cell_4/MatMul/ReadVariableOp2X
*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp*lstm_4/lstm_cell_4/MatMul_1/ReadVariableOp2
lstm_4/whilelstm_4/while:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í#
Ü
while_body_411838
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_411862_0:	-
while_lstm_cell_4_411864_0:	d)
while_lstm_cell_4_411866_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_411862:	+
while_lstm_cell_4_411864:	d'
while_lstm_cell_4_411866:	¢)while/lstm_cell_4/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0±
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_411862_0while_lstm_cell_4_411864_0while_lstm_cell_4_411866_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411778r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_4_411862while_lstm_cell_4_411862_0"6
while_lstm_cell_4_411864while_lstm_cell_4_411864_0"6
while_lstm_cell_4_411866while_lstm_cell_4_411866_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
Ò
Ò
H__inference_sequential_4_layer_call_and_return_conditional_losses_412376
lstm_4_input 
lstm_4_412363:	 
lstm_4_412365:	d
lstm_4_412367:	 
dense_4_412370:d
dense_4_412372:
identity¢dense_4/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCall
lstm_4/StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputlstm_4_412363lstm_4_412365lstm_4_412367*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_412274
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_412370dense_4_412372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_412086w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_4/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_4_input
ä8
Ç
while_body_413125
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0E
2while_lstm_cell_4_matmul_readvariableop_resource_0:	G
4while_lstm_cell_4_matmul_1_readvariableop_resource_0:	dB
3while_lstm_cell_4_biasadd_readvariableop_resource_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorC
0while_lstm_cell_4_matmul_readvariableop_resource:	E
2while_lstm_cell_4_matmul_1_readvariableop_resource:	d@
1while_lstm_cell_4_biasadd_readvariableop_resource:	¢(while/lstm_cell_4/BiasAdd/ReadVariableOp¢'while/lstm_cell_4/MatMul/ReadVariableOp¢)while/lstm_cell_4/MatMul_1/ReadVariableOp
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0
'while/lstm_cell_4/MatMul/ReadVariableOpReadVariableOp2while_lstm_cell_4_matmul_readvariableop_resource_0*
_output_shapes
:	*
dtype0¸
while/lstm_cell_4/MatMulMatMul0while/TensorArrayV2Read/TensorListGetItem:item:0/while/lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)while/lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp4while_lstm_cell_4_matmul_1_readvariableop_resource_0*
_output_shapes
:	d*
dtype0
while/lstm_cell_4/MatMul_1MatMulwhile_placeholder_21while/lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
while/lstm_cell_4/addAddV2"while/lstm_cell_4/MatMul:product:0$while/lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
(while/lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp3while_lstm_cell_4_biasadd_readvariableop_resource_0*
_output_shapes	
:*
dtype0¤
while/lstm_cell_4/BiasAddBiasAddwhile/lstm_cell_4/add:z:00while/lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
!while/lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :ì
while/lstm_cell_4/splitSplit*while/lstm_cell_4/split/split_dim:output:0"while/lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitx
while/lstm_cell_4/SigmoidSigmoid while/lstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_1Sigmoid while/lstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mulMulwhile/lstm_cell_4/Sigmoid_1:y:0while_placeholder_3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
while/lstm_cell_4/ReluRelu while/lstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_1Mulwhile/lstm_cell_4/Sigmoid:y:0$while/lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/add_1AddV2while/lstm_cell_4/mul:z:0while/lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdz
while/lstm_cell_4/Sigmoid_2Sigmoid while/lstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdo
while/lstm_cell_4/Relu_1Reluwhile/lstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/lstm_cell_4/mul_2Mulwhile/lstm_cell_4/Sigmoid_2:y:0&while/lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdr
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : ì
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:0while/lstm_cell_4/mul_2:z:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: x
while/Identity_4Identitywhile/lstm_cell_4/mul_2:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
while/Identity_5Identitywhile/lstm_cell_4/add_1:z:0^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdÍ

while/NoOpNoOp)^while/lstm_cell_4/BiasAdd/ReadVariableOp(^while/lstm_cell_4/MatMul/ReadVariableOp*^while/lstm_cell_4/MatMul_1/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"h
1while_lstm_cell_4_biasadd_readvariableop_resource3while_lstm_cell_4_biasadd_readvariableop_resource_0"j
2while_lstm_cell_4_matmul_1_readvariableop_resource4while_lstm_cell_4_matmul_1_readvariableop_resource_0"f
0while_lstm_cell_4_matmul_readvariableop_resource2while_lstm_cell_4_matmul_readvariableop_resource_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2T
(while/lstm_cell_4/BiasAdd/ReadVariableOp(while/lstm_cell_4/BiasAdd/ReadVariableOp2R
'while/lstm_cell_4/MatMul/ReadVariableOp'while/lstm_cell_4/MatMul/ReadVariableOp2V
)while/lstm_cell_4/MatMul_1/ReadVariableOp)while/lstm_cell_4/MatMul_1/ReadVariableOp: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
ê
ö
-__inference_sequential_4_layer_call_fn_412106
lstm_4_input
unknown:	
	unknown_0:	d
	unknown_1:	
	unknown_2:d
	unknown_3:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCalllstm_4_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*'
_read_only_resource_inputs	
*0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_sequential_4_layer_call_and_return_conditional_losses_412093o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
&
_user_specified_namelstm_4_input
À
Ì
H__inference_sequential_4_layer_call_and_return_conditional_losses_412316

inputs 
lstm_4_412303:	 
lstm_4_412305:	d
lstm_4_412307:	 
dense_4_412310:d
dense_4_412312:
identity¢dense_4/StatefulPartitionedCall¢lstm_4/StatefulPartitionedCallü
lstm_4/StatefulPartitionedCallStatefulPartitionedCallinputslstm_4_412303lstm_4_412305lstm_4_412307*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_412274
dense_4/StatefulPartitionedCallStatefulPartitionedCall'lstm_4/StatefulPartitionedCall:output:0dense_4_412310dense_4_412312*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_412086w
IdentityIdentity(dense_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
NoOpNoOp ^dense_4/StatefulPartitionedCall^lstm_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:ÿÿÿÿÿÿÿÿÿ: : : : : 2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2@
lstm_4/StatefulPartitionedCalllstm_4/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
í#
Ü
while_body_411645
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_strided_slice_1_0W
Swhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0-
while_lstm_cell_4_411669_0:	-
while_lstm_cell_4_411671_0:	d)
while_lstm_cell_4_411673_0:	
while_identity
while_identity_1
while_identity_2
while_identity_3
while_identity_4
while_identity_5
while_strided_slice_1U
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor+
while_lstm_cell_4_411669:	+
while_lstm_cell_4_411671:	d'
while_lstm_cell_4_411673:	¢)while/lstm_cell_4/StatefulPartitionedCall
7while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ¦
)while/TensorArrayV2Read/TensorListGetItemTensorListGetItemSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0while_placeholder@while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
element_dtype0±
)while/lstm_cell_4/StatefulPartitionedCallStatefulPartitionedCall0while/TensorArrayV2Read/TensorListGetItem:item:0while_placeholder_2while_placeholder_3while_lstm_cell_4_411669_0while_lstm_cell_4_411671_0while_lstm_cell_4_411673_0*
Tin

2*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_411630r
0while/TensorArrayV2Write/TensorListSetItem/indexConst*
_output_shapes
: *
dtype0*
value	B : 
*while/TensorArrayV2Write/TensorListSetItemTensorListSetItemwhile_placeholder_19while/TensorArrayV2Write/TensorListSetItem/index:output:02while/lstm_cell_4/StatefulPartitionedCall:output:0*
_output_shapes
: *
element_dtype0:éèÒM
while/add/yConst*
_output_shapes
: *
dtype0*
value	B :\
	while/addAddV2while_placeholderwhile/add/y:output:0*
T0*
_output_shapes
: O
while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :g
while/add_1AddV2while_while_loop_counterwhile/add_1/y:output:0*
T0*
_output_shapes
: Y
while/IdentityIdentitywhile/add_1:z:0^while/NoOp*
T0*
_output_shapes
: j
while/Identity_1Identitywhile_while_maximum_iterations^while/NoOp*
T0*
_output_shapes
: Y
while/Identity_2Identitywhile/add:z:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_3Identity:while/TensorArrayV2Write/TensorListSetItem:output_handle:0^while/NoOp*
T0*
_output_shapes
: 
while/Identity_4Identity2while/lstm_cell_4/StatefulPartitionedCall:output:1^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
while/Identity_5Identity2while/lstm_cell_4/StatefulPartitionedCall:output:2^while/NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx

while/NoOpNoOp*^while/lstm_cell_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 ")
while_identitywhile/Identity:output:0"-
while_identity_1while/Identity_1:output:0"-
while_identity_2while/Identity_2:output:0"-
while_identity_3while/Identity_3:output:0"-
while_identity_4while/Identity_4:output:0"-
while_identity_5while/Identity_5:output:0"6
while_lstm_cell_4_411669while_lstm_cell_4_411669_0"6
while_lstm_cell_4_411671while_lstm_cell_4_411671_0"6
while_lstm_cell_4_411673while_lstm_cell_4_411673_0"0
while_strided_slice_1while_strided_slice_1_0"¨
Qwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorSwhile_tensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : 2V
)while/lstm_cell_4/StatefulPartitionedCall)while/lstm_cell_4/StatefulPartitionedCall: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
: 
µ
Ã
while_cond_411982
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_411982___redundant_placeholder04
0while_while_cond_411982___redundant_placeholder14
0while_while_cond_411982___redundant_placeholder24
0while_while_cond_411982___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
ù	
Ï
lstm_4_while_cond_412639*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_412639___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_412639___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_412639___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_412639___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_411837
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_411837___redundant_placeholder04
0while_while_cond_411837___redundant_placeholder14
0while_while_cond_411837___redundant_placeholder24
0while_while_cond_411837___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
íJ

B__inference_lstm_4_layer_call_and_return_conditional_losses_413355

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	?
,lstm_cell_4_matmul_1_readvariableop_resource:	d:
+lstm_cell_4_biasadd_readvariableop_resource:	
identity¢"lstm_cell_4/BiasAdd/ReadVariableOp¢!lstm_cell_4/MatMul/ReadVariableOp¢#lstm_cell_4/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitl
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_413270*
condR
while_cond_413269*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ù	
Ï
lstm_4_while_cond_412488*
&lstm_4_while_lstm_4_while_loop_counter0
,lstm_4_while_lstm_4_while_maximum_iterations
lstm_4_while_placeholder
lstm_4_while_placeholder_1
lstm_4_while_placeholder_2
lstm_4_while_placeholder_3,
(lstm_4_while_less_lstm_4_strided_slice_1B
>lstm_4_while_lstm_4_while_cond_412488___redundant_placeholder0B
>lstm_4_while_lstm_4_while_cond_412488___redundant_placeholder1B
>lstm_4_while_lstm_4_while_cond_412488___redundant_placeholder2B
>lstm_4_while_lstm_4_while_cond_412488___redundant_placeholder3
lstm_4_while_identity
~
lstm_4/while/LessLesslstm_4_while_placeholder(lstm_4_while_less_lstm_4_strided_slice_1*
T0*
_output_shapes
: Y
lstm_4/while/IdentityIdentitylstm_4/while/Less:z:0*
T0
*
_output_shapes
: "7
lstm_4_while_identitylstm_4/while/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
µ
Ã
while_cond_411644
while_while_loop_counter"
while_while_maximum_iterations
while_placeholder
while_placeholder_1
while_placeholder_2
while_placeholder_3
while_less_strided_slice_14
0while_while_cond_411644___redundant_placeholder04
0while_while_cond_411644___redundant_placeholder14
0while_while_cond_411644___redundant_placeholder24
0while_while_cond_411644___redundant_placeholder3
while_identity
b

while/LessLesswhile_placeholderwhile_less_strided_slice_1*
T0*
_output_shapes
: K
while/IdentityIdentitywhile/Less:z:0*
T0
*
_output_shapes
: ")
while_identitywhile/Identity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: ::::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:-)
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd:

_output_shapes
: :

_output_shapes
:
íJ

B__inference_lstm_4_layer_call_and_return_conditional_losses_412068

inputs=
*lstm_cell_4_matmul_readvariableop_resource:	?
,lstm_cell_4_matmul_1_readvariableop_resource:	d:
+lstm_cell_4_biasadd_readvariableop_resource:	
identity¢"lstm_cell_4/BiasAdd/ReadVariableOp¢!lstm_cell_4/MatMul/ReadVariableOp¢#lstm_cell_4/MatMul_1/ReadVariableOp¢while;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskP
zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :ds
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
T0*
_output_shapes
:P
zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdR
zeros_1/packed/1Const*
_output_shapes
: *
dtype0*
value	B :dw
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
T0*
_output_shapes
:R
zeros_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
transpose/permConst*
_output_shapes
:*
dtype0*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿD
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Û
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿ´
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   à
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒ_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:é
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
shrink_axis_mask
!lstm_cell_4/MatMul/ReadVariableOpReadVariableOp*lstm_cell_4_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
lstm_cell_4/MatMulMatMulstrided_slice_2:output:0)lstm_cell_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#lstm_cell_4/MatMul_1/ReadVariableOpReadVariableOp,lstm_cell_4_matmul_1_readvariableop_resource*
_output_shapes
:	d*
dtype0
lstm_cell_4/MatMul_1MatMulzeros:output:0+lstm_cell_4/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
lstm_cell_4/addAddV2lstm_cell_4/MatMul:product:0lstm_cell_4/MatMul_1:product:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"lstm_cell_4/BiasAdd/ReadVariableOpReadVariableOp+lstm_cell_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
lstm_cell_4/BiasAddBiasAddlstm_cell_4/add:z:0*lstm_cell_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ]
lstm_cell_4/split/split_dimConst*
_output_shapes
: *
dtype0*
value	B :Ú
lstm_cell_4/splitSplit$lstm_cell_4/split/split_dim:output:0lstm_cell_4/BiasAdd:output:0*
T0*`
_output_shapesN
L:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd*
	num_splitl
lstm_cell_4/SigmoidSigmoidlstm_cell_4/split:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_1Sigmoidlstm_cell_4/split:output:1*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdu
lstm_cell_4/mulMullstm_cell_4/Sigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdf
lstm_cell_4/ReluRelulstm_cell_4/split:output:2*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_1Mullstm_cell_4/Sigmoid:y:0lstm_cell_4/Relu:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdx
lstm_cell_4/add_1AddV2lstm_cell_4/mul:z:0lstm_cell_4/mul_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
lstm_cell_4/Sigmoid_2Sigmoidlstm_cell_4/split:output:3*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdc
lstm_cell_4/Relu_1Relulstm_cell_4/add_1:z:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd
lstm_cell_4/mul_2Mullstm_cell_4/Sigmoid_2:y:0 lstm_cell_4/Relu_1:activations:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿdn
TensorArrayV2_1/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   ^
TensorArrayV2_1/num_elementsConst*
_output_shapes
: *
dtype0*
value	B :Å
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0%TensorArrayV2_1/num_elements:output:0*
_output_shapes
: *
element_dtype0*

shape_type0:éèÒF
timeConst*
_output_shapes
: *
dtype0*
value	B : c
while/maximum_iterationsConst*
_output_shapes
: *
dtype0*
valueB :
ÿÿÿÿÿÿÿÿÿT
while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : ÿ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0*lstm_cell_4_matmul_readvariableop_resource,lstm_cell_4_matmul_1_readvariableop_resource+lstm_cell_4_biasadd_readvariableop_resource*
T
2*
_lower_using_switch_merge(*
_num_original_outputs*L
_output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *%
_read_only_resource_inputs
	
*
_stateful_parallelism( *
bodyR
while_body_411983*
condR
while_cond_411982*K
output_shapes:
8: : : : :ÿÿÿÿÿÿÿÿÿd:ÿÿÿÿÿÿÿÿÿd: : : : : *
parallel_iterations 
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿd   Ö
"TensorArrayV2Stack/TensorListStackTensorListStackwhile:output:39TensorArrayV2Stack/TensorListStack/element_shape:output:0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
element_dtype0*
num_elementsh
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB:
ÿÿÿÿÿÿÿÿÿa
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*
shrink_axis_maske
transpose_1/permConst*
_output_shapes
:*
dtype0*!
valueB"          
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿd[
runtimeConst"/device:CPU:0*
_output_shapes
: *
dtype0*
valueB
 *    g
IdentityIdentitystrided_slice_3:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd½
NoOpNoOp#^lstm_cell_4/BiasAdd/ReadVariableOp"^lstm_cell_4/MatMul/ReadVariableOp$^lstm_cell_4/MatMul_1/ReadVariableOp^while*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : : 2H
"lstm_cell_4/BiasAdd/ReadVariableOp"lstm_cell_4/BiasAdd/ReadVariableOp2F
!lstm_cell_4/MatMul/ReadVariableOp!lstm_cell_4/MatMul/ReadVariableOp2J
#lstm_cell_4/MatMul_1/ReadVariableOp#lstm_cell_4/MatMul_1/ReadVariableOp2
whilewhile:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¶
'__inference_lstm_4_layer_call_fn_412742
inputs_0
unknown:	
	unknown_0:	d
	unknown_1:	
identity¢StatefulPartitionedCallé
StatefulPartitionedCallStatefulPartitionedCallinputs_0unknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd*%
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_lstm_4_layer_call_and_return_conditional_losses_411715o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : 22
StatefulPartitionedCallStatefulPartitionedCall:^ Z
4
_output_shapes"
 :ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0"¿L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¸
serving_default¤
I
lstm_4_input9
serving_default_lstm_4_input:0ÿÿÿÿÿÿÿÿÿ;
dense_40
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:À
¾
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
	_default_save_signature

	optimizer
loss

signatures"
_tf_keras_sequential
Ú
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator
cell

state_spec"
_tf_keras_rnn_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
C
0
1
 2
3
4"
trackable_list_wrapper
C
0
1
 2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
!non_trainable_variables

"layers
#metrics
$layer_regularization_losses
%layer_metrics
	variables
trainable_variables
regularization_losses
__call__
	_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ê
&trace_0
'trace_1
(trace_2
)trace_32ÿ
-__inference_sequential_4_layer_call_fn_412106
-__inference_sequential_4_layer_call_fn_412414
-__inference_sequential_4_layer_call_fn_412429
-__inference_sequential_4_layer_call_fn_412344À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z&trace_0z'trace_1z(trace_2z)trace_3
Ö
*trace_0
+trace_1
,trace_2
-trace_32ë
H__inference_sequential_4_layer_call_and_return_conditional_losses_412580
H__inference_sequential_4_layer_call_and_return_conditional_losses_412731
H__inference_sequential_4_layer_call_and_return_conditional_losses_412360
H__inference_sequential_4_layer_call_and_return_conditional_losses_412376À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z*trace_0z+trace_1z,trace_2z-trace_3
ÑBÎ
!__inference__wrapped_model_411563lstm_4_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
­
.iter

/beta_1

0beta_2
	1decay
2learning_ratememfmgmh mivjvkvlvm vn"
	optimizer
 "
trackable_list_wrapper
,
3serving_default"
signature_map
5
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
¹

4states
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ç
:trace_0
;trace_1
<trace_2
=trace_32ü
'__inference_lstm_4_layer_call_fn_412742
'__inference_lstm_4_layer_call_fn_412753
'__inference_lstm_4_layer_call_fn_412764
'__inference_lstm_4_layer_call_fn_412775Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z:trace_0z;trace_1z<trace_2z=trace_3
Ó
>trace_0
?trace_1
@trace_2
Atrace_32è
B__inference_lstm_4_layer_call_and_return_conditional_losses_412920
B__inference_lstm_4_layer_call_and_return_conditional_losses_413065
B__inference_lstm_4_layer_call_and_return_conditional_losses_413210
B__inference_lstm_4_layer_call_and_return_conditional_losses_413355Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 z>trace_0z?trace_1z@trace_2zAtrace_3
"
_generic_user_object
ø
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H_random_generator
I
state_size

kernel
recurrent_kernel
 bias"
_tf_keras_layer
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Jnon_trainable_variables

Klayers
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ì
Otrace_02Ï
(__inference_dense_4_layer_call_fn_413364¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zOtrace_0

Ptrace_02ê
C__inference_dense_4_layer_call_and_return_conditional_losses_413374¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zPtrace_0
 :d2dense_4/kernel
:2dense_4/bias
,:*	2lstm_4/lstm_cell_4/kernel
6:4	d2#lstm_4/lstm_cell_4/recurrent_kernel
&:$2lstm_4/lstm_cell_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
-__inference_sequential_4_layer_call_fn_412106lstm_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÿBü
-__inference_sequential_4_layer_call_fn_412414inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
ÿBü
-__inference_sequential_4_layer_call_fn_412429inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
-__inference_sequential_4_layer_call_fn_412344lstm_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_sequential_4_layer_call_and_return_conditional_losses_412580inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
H__inference_sequential_4_layer_call_and_return_conditional_losses_412731inputs"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 B
H__inference_sequential_4_layer_call_and_return_conditional_losses_412360lstm_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
 B
H__inference_sequential_4_layer_call_and_return_conditional_losses_412376lstm_4_input"À
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÐBÍ
$__inference_signature_wrapper_412399lstm_4_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
'__inference_lstm_4_layer_call_fn_412742inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
'__inference_lstm_4_layer_call_fn_412753inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
'__inference_lstm_4_layer_call_fn_412764inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
'__inference_lstm_4_layer_call_fn_412775inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«B¨
B__inference_lstm_4_layer_call_and_return_conditional_losses_412920inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«B¨
B__inference_lstm_4_layer_call_and_return_conditional_losses_413065inputs/0"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
©B¦
B__inference_lstm_4_layer_call_and_return_conditional_losses_413210inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
©B¦
B__inference_lstm_4_layer_call_and_return_conditional_losses_413355inputs"Õ
Ì²È
FullArgSpecB
args:7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaults

 
p 

 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
5
0
1
 2"
trackable_list_wrapper
5
0
1
 2"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
Ô
Xtrace_0
Ytrace_12
,__inference_lstm_cell_4_layer_call_fn_413391
,__inference_lstm_cell_4_layer_call_fn_413408¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zXtrace_0zYtrace_1

Ztrace_0
[trace_12Ó
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413440
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413472¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 zZtrace_0z[trace_1
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ÜBÙ
(__inference_dense_4_layer_call_fn_413364inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
÷Bô
C__inference_dense_4_layer_call_and_return_conditional_losses_413374inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
\	variables
]	keras_api
	^total
	_count"
_tf_keras_metric
^
`	variables
a	keras_api
	btotal
	ccount
d
_fn_kwargs"
_tf_keras_metric
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
,__inference_lstm_cell_4_layer_call_fn_413391inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
B
,__inference_lstm_cell_4_layer_call_fn_413408inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«B¨
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413440inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
«B¨
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413472inputsstates/0states/1"¾
µ²±
FullArgSpec3
args+(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaultsª 
annotationsª *
 
.
^0
_1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
.
b0
c1"
trackable_list_wrapper
-
`	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
%:#d2Adam/dense_4/kernel/m
:2Adam/dense_4/bias/m
1:/	2 Adam/lstm_4/lstm_cell_4/kernel/m
;:9	d2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/m
+:)2Adam/lstm_4/lstm_cell_4/bias/m
%:#d2Adam/dense_4/kernel/v
:2Adam/dense_4/bias/v
1:/	2 Adam/lstm_4/lstm_cell_4/kernel/v
;:9	d2*Adam/lstm_4/lstm_cell_4/recurrent_kernel/v
+:)2Adam/lstm_4/lstm_cell_4/bias/v
!__inference__wrapped_model_411563u 9¢6
/¢,
*'
lstm_4_inputÿÿÿÿÿÿÿÿÿ
ª "1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ£
C__inference_dense_4_layer_call_and_return_conditional_losses_413374\/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
(__inference_dense_4_layer_call_fn_413364O/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿd
ª "ÿÿÿÿÿÿÿÿÿÃ
B__inference_lstm_4_layer_call_and_return_conditional_losses_412920} O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 Ã
B__inference_lstm_4_layer_call_and_return_conditional_losses_413065} O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ³
B__inference_lstm_4_layer_call_and_return_conditional_losses_413210m ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 ³
B__inference_lstm_4_layer_call_and_return_conditional_losses_413355m ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿd
 
'__inference_lstm_4_layer_call_fn_412742p O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
'__inference_lstm_4_layer_call_fn_412753p O¢L
E¢B
41
/,
inputs/0ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿd
'__inference_lstm_4_layer_call_fn_412764` ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p 

 
ª "ÿÿÿÿÿÿÿÿÿd
'__inference_lstm_4_layer_call_fn_412775` ?¢<
5¢2
$!
inputsÿÿÿÿÿÿÿÿÿ

 
p

 
ª "ÿÿÿÿÿÿÿÿÿdÉ
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413440ý ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p 
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿd
EB

0/1/0ÿÿÿÿÿÿÿÿÿd

0/1/1ÿÿÿÿÿÿÿÿÿd
 É
G__inference_lstm_cell_4_layer_call_and_return_conditional_losses_413472ý ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p
ª "s¢p
i¢f

0/0ÿÿÿÿÿÿÿÿÿd
EB

0/1/0ÿÿÿÿÿÿÿÿÿd

0/1/1ÿÿÿÿÿÿÿÿÿd
 
,__inference_lstm_cell_4_layer_call_fn_413391í ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p 
ª "c¢`

0ÿÿÿÿÿÿÿÿÿd
A>

1/0ÿÿÿÿÿÿÿÿÿd

1/1ÿÿÿÿÿÿÿÿÿd
,__inference_lstm_cell_4_layer_call_fn_413408í ¢}
v¢s
 
inputsÿÿÿÿÿÿÿÿÿ
K¢H
"
states/0ÿÿÿÿÿÿÿÿÿd
"
states/1ÿÿÿÿÿÿÿÿÿd
p
ª "c¢`

0ÿÿÿÿÿÿÿÿÿd
A>

1/0ÿÿÿÿÿÿÿÿÿd

1/1ÿÿÿÿÿÿÿÿÿd½
H__inference_sequential_4_layer_call_and_return_conditional_losses_412360q A¢>
7¢4
*'
lstm_4_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ½
H__inference_sequential_4_layer_call_and_return_conditional_losses_412376q A¢>
7¢4
*'
lstm_4_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
H__inference_sequential_4_layer_call_and_return_conditional_losses_412580k ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
H__inference_sequential_4_layer_call_and_return_conditional_losses_412731k ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
-__inference_sequential_4_layer_call_fn_412106d A¢>
7¢4
*'
lstm_4_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_412344d A¢>
7¢4
*'
lstm_4_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_412414^ ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
-__inference_sequential_4_layer_call_fn_412429^ ;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ®
$__inference_signature_wrapper_412399 I¢F
¢ 
?ª<
:
lstm_4_input*'
lstm_4_inputÿÿÿÿÿÿÿÿÿ"1ª.
,
dense_4!
dense_4ÿÿÿÿÿÿÿÿÿ