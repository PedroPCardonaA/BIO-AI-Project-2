use rand::{seq::{IndexedRandom, IteratorRandom, SliceRandom}, Rng};
use structs::{depot::Depot, instance::Instance, patient::Patient};
use std::collections::{HashMap, HashSet};

mod structs;
mod utils;

fn main() {
    let instance = utils::parse_data::parse_data("src/data/train/train_0.json");
    
    let best_solution = evolutionary_algorithm(
        &instance,
        100,
        5000,
        5,
        0.2,
        1.2,
        100
    );

    plot_map(&best_solution, &instance.patients, &instance.depot);
    let _ = utils::create_file::save_solution_to_file(&best_solution, "solution.json");
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

fn generate_population_heuristic_with_workload(
    population_size: usize,
    instance: &Instance,
) -> Vec<Vec<Vec<usize>>> {
    let mut population = Vec::with_capacity(population_size);
    let patient_count = instance.patients.len();
    let nurse_count = instance.nurses.len();
    let mut rng = rand::thread_rng();

    for _ in 0..population_size {
        // Create a shuffled list of patient IDs (assumed to be 1-based).
        let mut patient_ids: Vec<usize> = (1..=patient_count).collect();
        patient_ids.shuffle(&mut rng);

        // Clone the nurse list to track each nurse's current load.
        let mut nurses = instance.nurses.clone();
        // Each solution is represented as a vector of routes (each route is a Vec of patient IDs).
        let mut solution = vec![Vec::new(); nurse_count];

        // First, assign one patient to each nurse (if available and if capacity allows).
        for i in 0..nurse_count {
            if let Some(patient) = patient_ids.pop() {
                {
                    let current_load = nurses[i].get_current_load();
                    let capacity = nurses[i].get_capacity();
                    if current_load < capacity {
                        solution[i].push(patient);
                        let new_load = current_load + 1;
                        nurses[i].set_current_load(new_load);
                    }
                }
            }
        }

        // For each remaining patient, choose the nurse that minimizes the balanced increase.
        while let Some(patient) = patient_ids.pop() {
            let mut best_nurse_index = None;
            let mut best_balanced_increase = f64::MAX;

            // Consider only nurses with available capacity.
            for (i, route) in solution.iter().enumerate() {
                let current_load = nurses[i].get_current_load();
                let capacity = nurses[i].get_capacity();
                if current_load >= capacity {
                    continue;
                }

                // Compute the extra travel time cost of appending the patient.
                let increase = if route.is_empty() {
                    // For an empty route, cost = depot -> patient + patient -> depot.
                    instance.travel_times[0][patient] + instance.travel_times[patient][0]
                } else {
                    // For a non-empty route, cost = (last patient -> new patient + new patient -> depot)
                    // minus (last patient -> depot) already accounted for.
                    let last_patient = *route.last().unwrap();
                    instance.travel_times[last_patient][patient]
                        + instance.travel_times[patient][0]
                        - instance.travel_times[last_patient][0]
                };

                // Instead of a fixed penalty, add the current load as a cost.
                let balanced_increase = increase + (current_load as f64);

                if balanced_increase < best_balanced_increase {
                    best_balanced_increase = balanced_increase;
                    best_nurse_index = Some(i);
                }
            }

            if let Some(i) = best_nurse_index {
                // Assign the patient to the chosen nurse.
                solution[i].push(patient);
                let new_load = nurses[i].get_current_load() + 1;
                nurses[i].set_current_load(new_load);
            } else {
                // If no nurse has available capacity (all reached capacity),
                // assign the patient to the nurse with the smallest overload.
                let mut min_overload_index = 0;
                let mut min_overload = f64::MAX;
                for (i, nurse) in nurses.iter().enumerate() {
                    let overload = nurse.get_current_load() as f64 - nurse.get_capacity() as f64;
                    if overload < min_overload {
                        min_overload = overload;
                        min_overload_index = i;
                    }
                }
                solution[min_overload_index].push(patient);
                let new_load = nurses[min_overload_index].get_current_load() + 1;
                nurses[min_overload_index].set_current_load(new_load);
            }
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
    let penalty_factor = 2.0; // Higher value means higher penalty
    let penalty_factor_time = 6.0; // Higher value means higher penalty

    // Calculate the total travel time for each nurse
    let mut nurses = instance.nurses.clone();
    for (nurse, route) in nurses.iter_mut().zip(solution.iter()) {
        let mut last_patient = 0; // The depot is the first patient

        //println!("New nurse");

        // Calculate the travel time and capacity for each patient in the route
        for patient_id in route {
            let mut wait_time = 0.0;
            let patient = &instance.patients[&patient_id.to_string()];
            // Print the nurse and patient ID
            //println!("Last patient: {:?}, Current patient: {:?}", last_patient, patient_id);

            // Calculate the travel time from the last patient to the current patient
            let travel_time = instance.travel_times[last_patient][*patient_id];

            // Add the travel time as a penalty, since then nurses will be penalized for traveling too much between patients
            total_penalty += travel_time * penalty_factor_time;

            // Check if the nurse visits the patient too early
            if patient.start_time > (nurse.get_current_travel_time() + travel_time) {
                wait_time = patient.start_time - (nurse.get_current_travel_time() + travel_time);
            }

            // Add the travel time to the nurse's current travel time
            nurse.set_current_travel_time(nurse.get_current_travel_time() + travel_time + patient.care_time + wait_time);

            // Check if the nurse visits the patient too late
            if patient.end_time < nurse.get_current_travel_time() {
                total_penalty += penalty_factor * (nurse.get_current_travel_time() - patient.end_time);
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

        // Check if the nurse returns to the depot too late
        if nurse.get_current_travel_time() > instance.depot.return_time {
            total_penalty += penalty_factor * (nurse.get_current_travel_time() - instance.depot.return_time);
        }

        // Add the nurse's travel time to the total travel time
        total_travel_time += nurse.get_current_travel_time();
    }

    total_travel_time + total_penalty
}


/// Selects one individual from the population using tournament selection.
///
/// # Arguments
///
/// * `population` - A reference to the population (each individual is a Vec<Vec<usize>>).
/// * `fitness` - A reference to a vector of fitness values corresponding to each individual.
///               Lower fitness is considered better.
/// * `tournament_size` - The number of individuals in each tournament.
///
/// # Returns
///
/// A clone of the selected individual.
pub fn tournament_selection(
    population: &Vec<Vec<Vec<usize>>>,
    fitness: &Vec<f64>,
    tournament_size: usize,
) -> Vec<Vec<usize>> {
    let mut rng = rand::rng();
    let pop_size = population.len();

    // Randomly select indices for the tournament.
    let mut best_index = None;
    for _ in 0..tournament_size {
        let idx = rng.random_range(0..pop_size);
        best_index = match best_index {
            Some(current_best) => {
                // Lower fitness is better.
                if fitness[idx] < fitness[current_best] {
                    Some(idx)
                } else {
                    Some(current_best)
                }
            }
            None => Some(idx),
        };
    }

    // Return the best individual from the tournament.
    population[best_index.unwrap()].clone()
}


pub fn exponential_rank_wheel_selection(
    population: &Vec<Vec<Vec<usize>>>,
    fitness: &Vec<f64>,
    lambda: f64,
) -> Vec<Vec<usize>> {
    // Create a vector of indices and sort them by fitness (ascending).
    let mut indices: Vec<usize> = (0..population.len()).collect();
    indices.sort_by(|&a, &b| fitness[a].partial_cmp(&fitness[b]).unwrap());

    // Compute exponential weights based on rank (best individual has rank 0).
    // weight = exp(-lambda * rank)
    let weights: Vec<f64> = indices
        .iter()
        .enumerate()
        .map(|(rank, _)| (-lambda * (rank as f64)).exp())
        .collect();

    // Calculate total weight.
    let total_weight: f64 = weights.iter().sum();

    // Build cumulative weights for the roulette wheel.
    let mut cumulative_weights = Vec::with_capacity(weights.len());
    let mut cumulative = 0.0;
    for w in &weights {
        cumulative += *w;
        cumulative_weights.push(cumulative);
    }

    // Generate a random number in [0, total_weight).
    let mut rng = rand::rng();
    let r: f64 = rng.random_range(0.0..total_weight);

    // Find the first rank where the cumulative weight exceeds r.
    let selected_rank = cumulative_weights
        .iter()
        .position(|&cw| cw >= r)
        .unwrap();
    let selected_index = indices[selected_rank];

    // Return the selected individual.
    population[selected_index].clone()
}



fn route_preserving_crossover(
    parent1: &Vec<Vec<usize>>, 
    parent2: &Vec<Vec<usize>>, 
    instance: &Instance
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut rng = rand::rng();
    let nurse_count = parent1.len();
    let patient_count = instance.patients.len();

    // Initialize children
    let mut child1 = vec![Vec::new(); nurse_count];
    let mut child2 = vec![Vec::new(); nurse_count];
    let mut used_patients: HashSet<usize> = HashSet::new();
    let mut assigned_nurses: HashSet<usize> = HashSet::new(); // Track assigned nurses

    // Step 1: Identify common routes (including different nurse indices)
    let mut route_map = HashMap::new();

    // Store routes in a map to find identical ones
    for nurse in 0..nurse_count {
        route_map.insert(parent1[nurse].clone(), nurse); // Store route -> nurse index
    }

    for nurse2 in 0..nurse_count {
        if let Some(&nurse1) = route_map.get(&parent2[nurse2]) {
            if !assigned_nurses.contains(&nurse1) && !assigned_nurses.contains(&nurse2) {
                // Copy the identical route to child1 and child2 at either index
                child1[nurse1] = parent1[nurse1].clone();
                child2[nurse1] = parent2[nurse2].clone();
                used_patients.extend(&child1[nurse1]);
                assigned_nurses.insert(nurse1);
                assigned_nurses.insert(nurse2);
            }
        }
    }

    // Step 2: Collect remaining unassigned patients
    let mut remaining_patients: Vec<usize> = (1..=patient_count)
        .filter(|p| !used_patients.contains(p))
        .collect();
    remaining_patients.shuffle(&mut rng);

    // Step 3: Assign remaining patients using an insertion heuristic
    for patient in remaining_patients {
        let mut best_nurse = 0;
        let mut best_increase = f64::MAX;

        for nurse in 0..nurse_count {
            let route = &child1[nurse];
            let last_patient = route.last().copied().unwrap_or(0); // 0 = depot
            let increase = instance.travel_times[last_patient][patient] 
                         + instance.travel_times[patient][0]; // Cost of adding patient

            if increase < best_increase {
                best_increase = increase;
                best_nurse = nurse;
            }
        }
        child1[best_nurse].push(patient);
        child2[best_nurse].push(patient);
    }

    // Step 4: Ensure nurse capacities are respected
    for nurse in 0..nurse_count {
        let mut total_demand: f64 = child1[nurse]
            .iter()
            .map(|p| instance.patients[&p.to_string()].demand) 
            .sum();

        while total_demand > instance.nurses[0].get_capacity() as f64 {
            if let Some(moved_patient) = child1[nurse].pop() {
                let new_nurse = rng.random_range(0..nurse_count);
                child1[new_nurse].push(moved_patient);
                child2[new_nurse].push(moved_patient);

                // Update total demand
                total_demand = child1[nurse]
                    .iter()
                    .map(|p| instance.patients[&p.to_string()].demand)
                    .sum();
            } else {
                break; // Prevent infinite loop if no more patients to move
            }
        }
    }

    (child1, child2)
}

pub fn mutate_relocate_patient(
    individual: &mut Vec<Vec<usize>>,
    mutation_probability: f64,
) {
    let mut rng = rand::rng();
    let num_nurses = individual.len();
    if num_nurses < 2 {
        return;
    }
    for i in 0..num_nurses {
        if !individual[i].is_empty() && rng.random::<f64>() < mutation_probability {
            let patient_index = rng.random_range(0..individual[i].len());
            let patient = individual[i].remove(patient_index);
            let other_nurses: Vec<usize> = (0..num_nurses).filter(|&j| j != i).collect();
            let target_nurse = *other_nurses.choose(&mut rng).unwrap();
            let insertion_index = rng.random_range(0..=individual[target_nurse].len());
            individual[target_nurse].insert(insertion_index, patient);
        }
    }
}

pub fn evolutionary_algorithm(
    instance: &Instance,
    population_size: usize,
    generations: usize,
    tournament_size: usize,
    mutation_probability: f64,
    lambda: f64,
    generation_to_print: usize,
) -> Vec<Vec<usize>> {
    // 1. Generate the initial population.
    let mut population = generate_population_heuristic_with_workload(population_size, instance);
    // Evaluate fitness for the initial population.
    let mut fitness_values: Vec<f64> = population
        .iter()
        .map(|individual| fitness(individual, instance))
        .collect();

    // Main loop: run for a fixed number of generations.
    for gen in 0..generations {
        let mut new_population = Vec::with_capacity(population_size);
        

        // Elitism: carry over the best individual to the next generation.
        let best_index = fitness_values
            .iter()
            .enumerate()
            .min_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
            .unwrap()
            .0;
        new_population.push(population[best_index].clone());

        // Generate new individuals until we fill the population.
        while new_population.len() < population_size {
            // Selection: choose two parents using tournament selection.
            let parent1 = tournament_selection(&population, &fitness_values, tournament_size);
            let parent2 = tournament_selection(&population, &fitness_values, tournament_size);

            // Crossover: perform a route-preserving crossover.
            let (mut child1, mut child2) = route_preserving_crossover(&parent1, &parent2, instance);

            // Mutation: apply mutation operator (relocate a patient) to each child.
            mutate_relocate_patient(&mut child1, mutation_probability);
            mutate_relocate_patient(&mut child2, mutation_probability);

            new_population.push(child1);
            if new_population.len() < population_size {
                new_population.push(child2);
            }
        }

        // Replace the old population with the new one and re-calculate fitness.
        population = new_population;
        fitness_values = population
            .iter()
            .map(|individual| fitness(individual, instance))
            .collect();

        // Print best fitness for this generation.
        let best_fit = fitness_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        if (gen) % generation_to_print == 0 {
            println!("Generation {}: Best fitness = {}", gen, best_fit);
        }
    }

    // Return the best solution from the final population.
    let best_index = fitness_values
        .iter()
        .enumerate()
        .min_by(|(_, &fit_a), (_, &fit_b)| fit_a.partial_cmp(&fit_b).unwrap())
        .unwrap()
        .0;
    population[best_index].clone()
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
