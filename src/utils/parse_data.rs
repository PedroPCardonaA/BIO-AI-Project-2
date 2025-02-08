use crate::structs::instance::Instance;
use std::fs::File;
use std::io::BufReader;

pub fn parse_data(path: &str) -> Instance {
    let file = File::open(path).expect("File not found");
    let reader = BufReader::new(file);
    let mut instance: Instance = serde_json::from_reader(reader).expect("Error while reading file");
    instance.add_nurses();
    instance
}
