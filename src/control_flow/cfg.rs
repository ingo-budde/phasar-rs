use crate::ir::IRDescription;

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ProgramPos<IR: IRDescription> {
    function: IR::Function,
    instruction: IR::Instruction,
}

pub struct ICFGEdge<IR: IRDescription> {
    current: ProgramPos<IR>,
    successor: ProgramPos<IR>,
}

pub trait CFG<IR: IRDescription> {
    fn get_function(&self, of: IR::Instruction) -> IR::Function;
    fn get_function_name(&self, of: IR::Function) -> String;
    // TODO: Can we somehow just return a string_view?
    fn get_local_successors(&self, of: IR::Instruction) -> Vec<IR::Instruction>;
    // TODO: really want to return impl Iterator<Item=IR::Instruction>
    fn get_instructions(&self, of: IR::Instruction) -> Vec<IR::Instruction>;
    fn get_call_instructions(&self, of: IR::Instruction) -> Vec<IR::Instruction>;
    fn get_starting_point(&self, of: IR::Function) -> Option<IR::Instruction>;
    fn get_exit_points(&self, of: IR::Function) -> Vec<IR::Instruction>;

    fn is_call_site(&self, inst: IR::Instruction) -> bool;
    fn is_exit_point(&self, inst: IR::Instruction) -> bool;
    fn is_starting_point(&self, inst: IR::Instruction) -> bool;
}

pub trait ICFG<IR: IRDescription>: CFG<IR> {
    // TODO: again, really want to return impl Iterator<Item=IR::Function>
    fn get_callees(&self, of_call_at: IR::Instruction) -> Vec<IR::Function>;
    fn get_successors(&self, of: ProgramPos<IR>) -> Vec<ProgramPos<IR>>;
}
