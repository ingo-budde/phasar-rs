pub trait IRDescription {
    type Instruction: Copy + Eq;
    type Value: Copy + Eq;
    type Function: Copy + Eq;
    type GlobalVariable: Copy + Eq;
}
