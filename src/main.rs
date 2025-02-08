use rand::{seq::{IndexedRandom, IteratorRandom, SliceRandom}, Rng};
use structs::{depot::Depot, instance::Instance, patient::Patient};
use std::collections::{HashMap, HashSet};

mod structs;
mod utils;

fn main() {
    let instance = utils::parse_data::parse_data("src/data/train/train_0.json");
    let population = generate_population_heuristic(10, &instance);
    let parent_1: &Vec<Vec<usize>> = &population[0];

    // Calculate the fitness of the first solution
    let fitness_value = fitness(parent_1, &instance);
    println!("Fitness value: {:?}", fitness_value);

    utils::create_file::save_solution_to_file(parent_1, "parent_1.json").expect("Failed to save parent_1 to file");
    plot_map(parent_1, &instance.patients, &instance.depot);
}

// Not used, better to use the heuristic approach
fn generate_population(population_size: usize, instance: &Instance) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::new();
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();
    
    for _ in 0..population_size {
        let mut patients: Vec<usize> = (1..(patient_count + 1)).collect();
        patients.shuffle(&mut rng);

        let mut solution = vec![Vec::new(); nurse_count];

        // Randomly distribute patients to nurses
        for &patient in &patients {
            let nurse_index = rng.random_range(0..nurse_count);
            solution[nurse_index].push(patient);
        }

        population.push(solution);
    }
    
    population
}

use std::f64;

// Assume Instance, Patient, Depot, and Nurse are defined as in your project.

fn generate_population_heuristic(population_size: usize, instance: &Instance) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::with_capacity(population_size);
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::rng();
    
    // Parameter to penalize nurses that already have many patients.
    let load_penalty: f64 = 1.0; // Tune this value as needed.
    
    for _ in 0..population_size {
        // Create a shuffled list of patient IDs.
        let mut patient_ids: Vec<usize> = (1..=patient_count).collect();
        patient_ids.shuffle(&mut rng);
        
        // Each solution is a vector of routes (each route is a vector of patient IDs)
        // and each nurse's route starts and ends at the depot (index 0).
        let mut solution = vec![Vec::new(); nurse_count];
        
        // First, ensure that every nurse gets one patient if possible.
        for i in 0..nurse_count {
            if let Some(patient) = patient_ids.pop() {
                solution[i].push(patient);
            }
        }
        
        // For the remaining patients, assign each to the nurse that minimizes the balanced cost.
        while let Some(patient) = patient_ids.pop() {
            let mut best_nurse_index = 0;
            let mut best_balanced_increase = f64::MAX;
            
            for (i, route) in solution.iter().enumerate() {
                // Calculate the extra travel time of appending this patient.
                // If the route is empty (should not occur now because of the initial assignment),
                // use depot -> patient + patient -> depot.
                let increase = if route.is_empty() {
                    instance.travel_times[0][patient] + instance.travel_times[patient][0]
                } else {
                    // For a non-empty route, the additional cost is:
                    // travel time from the last patient in the route to the new patient,
                    // plus travel time from the new patient back to the depot,
                    // minus the current travel time from the last patient to the depot.
                    let last_patient = *route.last().unwrap();
                    instance.travel_times[last_patient][patient] 
                        + instance.travel_times[patient][0] 
                        - instance.travel_times[last_patient][0]
                };
                
                // Add a penalty proportional to the current number of patients in the nurse's route.
                let balanced_increase = increase + load_penalty * (route.len() as f64);
                
                if balanced_increase < best_balanced_increase {
                    best_balanced_increase = balanced_increase;
                    best_nurse_index = i;
                }
            }
            // Assign the patient to the nurse with the minimal balanced cost.
            solution[best_nurse_index].push(patient);
        }
        
        population.push(solution);
    }
    
    population
}

fn edge_crossover(parent1: &Vec<Vec<usize>>, parent2: &Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();
    let nurse_count = parent1.len();
    
    // Flatten parents into ordered lists of patients, while keeping nurse structure
    let p1_flat: Vec<usize> = parent1.iter().flatten().cloned().collect();
    let p2_flat: Vec<usize> = parent2.iter().flatten().cloned().collect();
    let patient_count = p1_flat.len();

    // Build adjacency edge map
    let mut edge_map: HashMap<usize, HashSet<usize>> = HashMap::new();
    
    for (p1, p2) in [(&p1_flat, &p2_flat), (&p2_flat, &p1_flat)].iter() {
        for i in 0..patient_count {
            let current = p1[i];
            let left = if i == 0 { p1[patient_count - 1] } else { p1[i - 1] };
            let right = if i == patient_count - 1 { p1[0] } else { p1[i + 1] };
            
            edge_map.entry(current).or_insert_with(HashSet::new).insert(left);
            edge_map.entry(current).or_insert_with(HashSet::new).insert(right);
        }
    }

    // Generate offspring as a valid permutation
    let mut offspring = Vec::new();
    let mut remaining: HashSet<usize> = p1_flat.iter().cloned().collect();
    
    let mut current = *p1_flat.choose(&mut rng).unwrap();
    offspring.push(current);
    remaining.remove(&current);

    while !remaining.is_empty() {
        // Remove current patient from all adjacency lists
        for neighbors in edge_map.values_mut() {
            neighbors.remove(&current);
        }

        // Choose the next patient
        let next = if let Some(neighbors) = edge_map.get(&current) {
            if !neighbors.is_empty() {
                // Prefer neighbors with fewer connections
                let mut sorted_neighbors: Vec<&usize> = neighbors.iter().collect();
                sorted_neighbors.sort_by_key(|n| edge_map.get(n).map_or(0, |s| s.len()));
                Some(*sorted_neighbors[0])
            } else {
                None
            }
        } else {
            None
        };

        // If no valid neighbor, pick randomly from remaining
        current = next.unwrap_or_else(|| *remaining.iter().choose(&mut rng).unwrap());
        offspring.push(current);
        remaining.remove(&current);
    }

    // **Redistribute offspring into nurses using parent1's structure**
    let mut index = 0;
    let distribution: Vec<usize> = parent1.iter().map(|n| n.len()).collect();
    let mut new_solution = vec![Vec::new(); nurse_count];

    for (i, &count) in distribution.iter().enumerate() {
        new_solution[i] = offspring[index..index + count].to_vec();
        index += count;
    }

    new_solution
}

/*  
    Fitness function
    A lower fitness value is better, so we can use the total travel time as the fitness value.
    Penalize solutions that exceed the maximum number of patients per nurse
    Should also penalize solutions where a nurse's capacity is exceeded
    If a patient is visited outside of their time window, penalize the solution
*/
fn fitness(solution: &Vec<Vec<usize>>, instance: &Instance) -> f64 {
    let mut total_travel_time = 0.0;
    let mut total_penalty = 0.0;
    let penalty_factor = 100.0; // Higher value means higher penalty

    // Calculate the total travel time for each nurse
    let mut nurses = instance.nurses.clone();
    for (nurse, route) in nurses.iter_mut().zip(solution.iter()) {
        let mut last_patient = 0; // The depot is the first patient

        //println!("New nurse");

        // Calculate the travel time and capacity for each patient in the route
        for patient_id in route {
            let patient = &instance.patients[&patient_id.to_string()];
            // Print the nurse and patient ID
            //println!("Last patient: {:?}, Current patient: {:?}", last_patient, patient_id);

            // Calculate the travel time from the last patient to the current patient
            let travel_time = instance.travel_times[last_patient][*patient_id];

            // Check if the nurse visits the patient too early
            if patient.start_time > nurse.get_current_total_time() + travel_time {
                total_penalty += penalty_factor * (patient.start_time - nurse.get_current_total_time() - travel_time);
            }

            // Add the travel time to the nurse's current travel time
            nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time + patient.care_time);

            // Check if the nurse visits the patient too late
            if patient.end_time < nurse.get_current_total_time() {
                total_penalty += penalty_factor * (nurse.get_current_total_time() - patient.end_time);
            }

            // Add the patient's demand to the nurse's current load
            nurse.set_current_load(nurse.get_current_load() + patient.demand as u32);

            // Set the current patient as the last patient
            last_patient = *patient_id;
        }

        // Add the travel time from the last patient to the depot
        let travel_time = instance.travel_times[last_patient][0];
        nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time);

        // Check if the nurses capacity is exceeded
        if nurse.get_current_load() as f64 > nurse.get_capacity() as f64 {
            total_penalty += penalty_factor * (nurse.get_current_load() as f64 - nurse.get_capacity() as f64);
        }

        // Add the nurse's travel time to the total travel time
        total_travel_time += nurse.get_current_travel_time();
    }

    total_travel_time + total_penalty
}

use plotters::{coord::types::RangedCoordf64, prelude::*};
use std::f64::consts::PI;

pub fn plot_map(solution: &Vec<Vec<usize>>, patients: &HashMap<String, Patient>, depot: &Depot) {
    let output_path = "solution.png";
    let root = BitMapBackend::new(output_path, (900, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Scale factor for the distances.
    let scale_factor = 1.5;

    // Helper function: scales a point relative to the depot.
    fn scale_point(x: f64, y: f64, depot: &Depot, factor: f64) -> (f64, f64) {
        (depot.x_coord + (x - depot.x_coord) * factor,
         depot.y_coord + (y - depot.y_coord) * factor)
    }

    // Compute scaled bounds for the chart.
    let (scaled_depot_x, scaled_depot_y) = (depot.x_coord, depot.y_coord);
    let mut min_x = scaled_depot_x;
    let mut max_x = scaled_depot_x;
    let mut min_y = scaled_depot_y;
    let mut max_y = scaled_depot_y;
    for patient in patients.values() {
        let (scaled_x, scaled_y) = scale_point(patient.x_coord, patient.y_coord, depot, scale_factor);
        if scaled_x < min_x { min_x = scaled_x; }
        if scaled_x > max_x { max_x = scaled_x; }
        if scaled_y < min_y { min_y = scaled_y; }
        if scaled_y > max_y { max_y = scaled_y; }
    }

    let mut chart = ChartBuilder::on(&root)
        .caption("Nurse Routing Solution", ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Generate unique colors for each nurse using HSL color space.
    let num_nurses = solution.len();
    let colors: Vec<RGBColor> = (0..num_nurses)
        .map(|i| {
            let hue = 360.0 * (i as f64 / num_nurses as f64);
            let (r, g, b) = hsl_to_rgb(hue, 0.8, 0.5);
            RGBColor(r, g, b)
        })
        .collect();

    // Draw the depot.
    let depot_point = scale_point(depot.x_coord, depot.y_coord, depot, scale_factor);
    chart
        .draw_series(std::iter::once(Circle::new(depot_point, 5, BLACK.filled())))
        .unwrap();

    // For each nurse, draw its route.
    for (nurse_id, route) in solution.iter().enumerate() {
        let color = colors[nurse_id];

        // Build the path: start at the depot, then visit each patient (scaled), and return to the depot.
        let mut path_points = vec![depot_point];
        for patient_id in route {
            if let Some(patient) = patients.get(&patient_id.to_string()) {
                let scaled_coords = scale_point(patient.x_coord, patient.y_coord, depot, scale_factor);
                path_points.push(scaled_coords);
            }
        }
        path_points.push(depot_point);

        // Draw the route as a line.
        chart
            .draw_series(LineSeries::new(path_points.iter().copied(), &color))
            .unwrap();

        // Draw arrows along the route.
        for window in path_points.windows(2) {
            if let [start, end] = *window {
                let angle = ((end.1 - start.1).atan2(end.0 - start.0)).to_degrees();
                draw_arrow(&mut chart, start, end, angle, &color);
            }
        }

        // Draw patient markers with a smaller radius.
        // Skip the first and last points (which are the depot) when drawing patient markers.
        for &point in path_points.iter().skip(1).take(path_points.len() - 2) {
            chart
                .draw_series(std::iter::once(Circle::new(point, 3, color.filled())))
                .unwrap();
        }
    }

    // Draw a legend in the top-right corner.
    let legend_x = max_x - (max_x - min_x) * 0.2;
    let legend_y = max_y - (max_y - min_y) * 0.05;
    for (i, color) in colors.iter().enumerate() {
        let legend_text = format!("Nurse {}", i + 1);
        chart.draw_series(std::iter::once(Text::new(
            legend_text,
            (legend_x, legend_y - i as f64 * 10.0),
            TextStyle::from(("sans-serif", 15).into_font()).color(color),
        )))
        .unwrap();
    }

    root.present().unwrap();
    println!("Solution diagram saved as {}", output_path);
}

/// Convert HSL to RGB.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Draw a small arrow at the end of a route segment.
fn draw_arrow(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    start: (f64, f64),
    end: (f64, f64),
    angle: f64,
    color: &RGBColor,
) {
    // Reduced arrow size.
    let arrow_length = 2.0;
    let angle_rad = angle.to_radians();

    let arrow_x1 = end.0 - arrow_length * (angle_rad + PI / 6.0).cos();
    let arrow_y1 = end.1 - arrow_length * (angle_rad + PI / 6.0).sin();
    let arrow_x2 = end.0 - arrow_length * (angle_rad - PI / 6.0).cos();
    let arrow_y2 = end.1 - arrow_length * (angle_rad - PI / 6.0).sin();

    chart
        .draw_series(LineSeries::new(vec![end, (arrow_x1, arrow_y1)], color))
        .unwrap();
    chart
        .draw_series(LineSeries::new(vec![end, (arrow_x2, arrow_y2)], color))
        .unwrap();
}
