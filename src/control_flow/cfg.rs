use crate::ir::IRDescription;

pub struct ProgramPos<IR: IRDescription> {
    function: IR::Function,
    instruction: IR::Instruction,
}

trait CFG<IR: IRDescription> {
    // TODO: implement
    fn get_function(&self, of: IR::Instruction) -> IR::Function;
    fn get_local_successors(&self, of: IR::Instruction) -> Vec<IR::Instruction>;
    // TODO: really want to return impl Iterator<Item=IR::Instruction>
    fn get_instructions(&self, of: IR::Instruction) -> Vec<IR::Instruction>;
    fn get_call_instructions(&self, of: IR::Instruction) -> Vec<IR::Instruction>;
}

trait ICFG<IR: IRDescription>: CFG<IR> {
    // TODO: again, really want to return impl Iterator<Item=IR::Function>
    fn get_callees(&self, of_call_at: IR::Instruction) -> Vec<IR::Function>;
    fn get_successors(&self, of: ProgramPos<IR>) -> Vec<ProgramPos<IR>>;
}
