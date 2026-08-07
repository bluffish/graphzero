use graphs_compiler::ir::{
    AddressSpace, DType, DefSite, Device, IrType, ScalarDType, TokenKind, Value,
};

const TENSOR_VALUE_TOKEN: u16 = 4;
const SCALAR_VALUE_TOKEN: u16 = 5;
const INDEX_VALUE_TOKEN: u16 = 6;
const BOOL_VALUE_TOKEN: u16 = 7;
const BUFFER_VALUE_TOKEN: u16 = 8;
const STATE_VALUE_TOKEN: u16 = 9;
const CONTROL_TOKEN_VALUE_TOKEN: u16 = 10;
const MEMORY_TOKEN_VALUE_TOKEN: u16 = 11;
const STATE_TOKEN_VALUE_TOKEN: u16 = 12;
const TUPLE_VALUE_TOKEN: u16 = 13;
const LIST_VALUE_TOKEN: u16 = 14;
const NONE_VALUE_TOKEN: u16 = 15;

pub(super) fn value_token(ty: &IrType) -> u16 {
    match ty {
        IrType::Tensor(_) => TENSOR_VALUE_TOKEN,
        IrType::Scalar(_) => SCALAR_VALUE_TOKEN,
        IrType::Index => INDEX_VALUE_TOKEN,
        IrType::Bool => BOOL_VALUE_TOKEN,
        IrType::Buffer(_) => BUFFER_VALUE_TOKEN,
        IrType::State { .. } => STATE_VALUE_TOKEN,
        IrType::Token(TokenKind::Control) => CONTROL_TOKEN_VALUE_TOKEN,
        IrType::Token(TokenKind::Memory) => MEMORY_TOKEN_VALUE_TOKEN,
        IrType::Token(TokenKind::State) => STATE_TOKEN_VALUE_TOKEN,
        IrType::Tuple(_) => TUPLE_VALUE_TOKEN,
        IrType::List(_) => LIST_VALUE_TOKEN,
        IrType::None => NONE_VALUE_TOKEN,
    }
}

pub(super) fn value_attrs(value: &Value, ty: &IrType) -> [f32; 8] {
    let mut attrs = [0.0; 8];
    attrs[0] = count_feature(value.uses.len());
    attrs[7] = match value.def {
        DefSite::ModuleTensor { .. } => 0.25,
        DefSite::ComponentInput { .. } => 0.5,
        DefSite::BodyArg { .. } => 0.75,
        DefSite::NodeResult { .. } => 1.0,
    };
    match ty {
        IrType::Tensor(tensor) => {
            let tensor = tensor.view();
            attrs[1] = count_feature(tensor.shape.rank());
            attrs[2] = magnitude_feature(tensor.shape.element_count());
            attrs[3] = dtype_feature(tensor.dtype);
            attrs[4] = bool_feature(matches!(tensor.device, Device::Target { .. }));
            attrs[5] = bool_feature(is_contiguous(
                tensor.shape.as_ref(),
                tensor.layout.view().strides,
            ));
            attrs[6] = magnitude_feature(tensor.layout.view().storage_offset);
        }
        IrType::Scalar(dtype) => attrs[3] = scalar_dtype_feature(*dtype),
        IrType::Buffer(buffer) => {
            let buffer = buffer.view();
            attrs[1] = count_feature(buffer.shape.rank());
            attrs[2] = magnitude_feature(buffer.shape.element_count());
            attrs[3] = dtype_feature(buffer.element);
            attrs[4] = address_space_feature(buffer.address_space);
            attrs[5] = bool_feature(is_contiguous(
                buffer.shape.as_ref(),
                buffer.layout.view().strides,
            ));
            attrs[6] = magnitude_feature(buffer.layout.view().storage_offset);
        }
        IrType::Tuple(values) | IrType::List(values) => {
            attrs[1] = count_feature(values.len());
        }
        IrType::State { .. } | IrType::Token(_) | IrType::Index | IrType::Bool | IrType::None => {}
    }
    attrs
}

fn scalar_dtype_feature(dtype: ScalarDType) -> f32 {
    let code = match dtype {
        ScalarDType::Bool => 1,
        ScalarDType::Uint8 => 2,
        ScalarDType::Int8 => 3,
        ScalarDType::Uint16 => 4,
        ScalarDType::Int16 => 5,
        ScalarDType::Uint32 => 6,
        ScalarDType::Int32 => 7,
        ScalarDType::Uint64 => 8,
        ScalarDType::Int64 => 9,
        ScalarDType::Fp16 => 10,
        ScalarDType::Bf16 => 11,
        ScalarDType::Fp32 => 12,
        ScalarDType::Fp64 => 13,
    };
    code as f32 / 14.0
}

fn dtype_feature(dtype: &DType) -> f32 {
    match dtype {
        DType::Scalar(dtype) => scalar_dtype_feature(*dtype),
        DType::Quantized(_) => 1.0,
    }
}

fn address_space_feature(address_space: AddressSpace) -> f32 {
    let code = match address_space {
        AddressSpace::Host => 0,
        AddressSpace::DeviceGlobal => 1,
        AddressSpace::Unified => 2,
        AddressSpace::DeviceShared => 3,
        AddressSpace::DeviceLocal => 4,
        AddressSpace::Constant => 5,
    };
    code as f32 / 5.0
}

fn is_contiguous(shape: &[u64], strides: &[u64]) -> bool {
    let mut expected = 1_u64;
    for (extent, stride) in shape.iter().zip(strides).rev() {
        if *extent > 1 && *stride != expected {
            return false;
        }
        expected = expected.saturating_mul(*extent);
    }
    true
}

fn count_feature(count: usize) -> f32 {
    let count = count as f32;
    count / (count + 1.0)
}

fn magnitude_feature(value: u64) -> f32 {
    (value as f64).ln_1p() as f32 / (u64::MAX as f64).ln_1p() as f32
}

const fn bool_feature(value: bool) -> f32 {
    if value { 1.0 } else { 0.0 }
}
