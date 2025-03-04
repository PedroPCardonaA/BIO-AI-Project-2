use crate::structs::instance::Instance;
use std::fs::File;
use std::io::{Write, BufWriter};

/// Saves a textual representation of the solution to a file.
/// 
/// This function writes a plain text summary of the solution to the specified file path. The summary includes:
/// - The instance details such as instance name, number of nurses, nurse capacity, and depot return time.
/// - For each nurse, the route information including travel time, covered demand, and the sequence of visited patients.
/// - The benchmark objective value and the computed total duration of the solution.
/// 
/// # Parameters:
/// - `path`: A string slice that specifies the file path where the solution will be saved.
/// - `solution`: A reference to the solution, represented as a vector of routes (each route is a vector of patient IDs).
/// - `instance`: A reference to the instance for which the solution was computed, containing all necessary configuration data.
/// 
/// # Returns:
/// This function returns `()` on success; it will panic if an error occurs during file creation or writing.
pub fn save_textual_solution_to_file(path: &str, solution: &Vec<Vec<usize>>, instance: &Instance) {
    let file = File::create(path).expect("Unable to create file");
    let mut writer = BufWriter::new(file);
    writeln!(
        writer,
        "Instance name: {}",
        instance.instance_name
    ).expect("Error writing instance name");

    writeln!(
        writer,
        "Number of nurses: {}",
        instance.nbr_nurses
    ).expect("Error writing number of nurses");

    writeln!(
        writer,
        "Nurse capacity: {}",
        instance.capacity_nurse
    ).expect("Error writing nurse capacity");

    writeln!(
        writer,
        "Depot return time: {}",
        instance.depot.return_time
    ).expect("Error writing depot return time");

    for _ in 0..80 {
        write!(writer, "-").expect("Error writing separator");
    }
    writeln!(writer).expect("Error writing separator");

    let mut total_travel_time = 0.0;

    for (nurse_idx, route) in solution.iter().enumerate() {
        let mut travel_time = 0.0;
        let mut covered_demand = 0.0;
        let mut current_time = 0.0;

        let mut last_loc = 0; 
        let mut patient_sequence = vec![format!("D({:.2})", current_time)];

        if !route.is_empty() {
            for &patient_id in route {
                let distance = instance.travel_times[last_loc][patient_id];
                travel_time += distance;
                let arrival_time = current_time + distance;
                let patient_key = patient_id.to_string();
                let patient = instance
                    .patients
                    .get(&patient_key)
                    .expect("Patient not found in instance");
                let start_of_service = if arrival_time < patient.start_time {
                    patient.start_time
                } else {
                    arrival_time
                };
                let leave_time = start_of_service + patient.care_time;
                current_time = leave_time;
                covered_demand += patient.demand;
                let patient_str = format!(
                    "P{}({:.2}-{:.2})[{:.2}-{:.2}]",
                    patient_id,
                    start_of_service,
                    leave_time,
                    patient.start_time,
                    patient.end_time
                );

                patient_sequence.push(patient_str);
                last_loc = patient_id;
            }
            let return_distance = instance.travel_times[last_loc][0];
            travel_time += return_distance;
            current_time += return_distance;
        }
        patient_sequence.push(format!("D({:.2})", current_time));
        total_travel_time += travel_time;
        writeln!(
            writer,
            "nurse #{}\t{:.2}\t{:.2}\t{}",
            nurse_idx + 1,
            travel_time,
            covered_demand,
            patient_sequence.join(" -> ")
        ).expect("Error writing nurse route");
    }
    for _ in 0..80 {
        write!(writer, "-").expect("Error writing separator");
    }
    writeln!(writer).expect("Error writing separator");
    writeln!(
        writer,
        "Benchmark objective value: {:.2}",
        instance.benchmark
    ).expect("Error writing benchmark objective value");
    writeln!(
        writer,
        "Objective value (total duration): {:.2}",
        total_travel_time
    ).expect("Error writing total duration");
}
