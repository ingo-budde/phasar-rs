use std::fs::File;
use std::path::PathBuf;
use log::LevelFilter;
use clap_derive::Parser;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct CommandLineArgs {
    #[arg(short, long, action = clap::ArgAction::Count)]
    pub verbose: u8,

    #[arg(short, long)]
    pub quiet: bool,

    #[arg(short, long)]
    pub log_file: Option<PathBuf>,
}
impl CommandLineArgs {
    fn get_log_level(&self) -> LevelFilter {
        if self.quiet { return LevelFilter::Off }
        match self.verbose {
            0 => LevelFilter::Info,
            1 => LevelFilter::Debug,
            _ => LevelFilter::Trace,
        }
    }
    pub fn init_logger(&self) {
        use simplelog::*;
        let log_level = self.get_log_level();
        if let Some(log_file) = &self.log_file {
            CombinedLogger::init(
                vec![
                    TermLogger::new(log_level, Config::default(), TerminalMode::Mixed, ColorChoice::Auto),
                    WriteLogger::new(LevelFilter::Info, Config::default(), File::create(log_file).unwrap()),
                ]
            ).unwrap();
        } else {
            TermLogger::init(log_level, Config::default(), TerminalMode::Mixed, ColorChoice::Auto).unwrap();
        }
    }
}