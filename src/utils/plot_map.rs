use plotters::{coord::types::RangedCoordf64, prelude::*};
use std::{collections::HashMap, f64::consts::PI};

use crate::structs::{depot::Depot, patient::Patient};

/// Plots the nurse routing solution on a map and saves it as a PNG image to the specified output path.
/// 
/// This function generates a visual representation of the solution by plotting the depot, patient locations,
/// and nurse routes. Patient coordinates are scaled relative to the depot using a scale factor. Unique colors
/// are generated for each nurse route via an HSL-to-RGB conversion, and arrows are drawn along the routes
/// to indicate direction. A legend is added on the right side of the plot to identify each nurse's route.
pub fn plot_map_with_path(
    solution: &Vec<Vec<usize>>,
    patients: &HashMap<String, Patient>,
    depot: &Depot,
    output_path: &str,
) {
    // Create a 1200x1200 canvas.
    let root = BitMapBackend::new(output_path, (1400, 1200)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    // Split the drawing area: left 80% for the map, right 20% for the legend.
    let (map_area, legend_area) = root.split_horizontally(1400.0*0.875);

    let scale_factor = 1.5;

    // Helper function to scale coordinates relative to the depot.
    fn scale_point(
        x: f64,
        y: f64,
        depot: &crate::structs::depot::Depot,
        factor: f64,
    ) -> (f64, f64) {
        (
            depot.x_coord + (x - depot.x_coord) * factor,
            depot.y_coord + (y - depot.y_coord) * factor,
        )
    }

    // Compute the scaled bounds for the chart.
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

    // Build and configure the chart on the left 80% map area.
    let mut chart = ChartBuilder::on(&map_area)
        .caption("Nurse Routing Solution", ("sans-serif", 30))
        .margin(15)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(min_x..max_x, min_y..max_y)
        .unwrap();
    chart.configure_mesh().draw().unwrap();

    // Generate unique colors for each nurse route using the HSL color space.
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

    // Draw each nurse's route.
    for (nurse_id, route) in solution.iter().enumerate() {
        let color = colors[nurse_id];
        // Build the route path: start at the depot, visit each patient, and return to the depot.
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

        // Draw arrows along the route to indicate direction.
        for window in path_points.windows(2) {
            if let [start, end] = *window {
                let angle = ((end.1 - start.1).atan2(end.0 - start.0)).to_degrees();
                draw_arrow(&mut chart, end, angle, &color);
            }
        }

        // Draw patient markers (excluding the depot).
        for &point in path_points.iter().skip(1).take(path_points.len() - 2) {
            chart
                .draw_series(std::iter::once(Circle::new(point, 3, color.filled())))
                .unwrap();
        }
    }

    // Draw the legend in the right 20% area.
    legend_area.fill(&WHITE).unwrap();
    // Use pixel coordinates within the legend_area.
    let legend_origin = (10, 10); // starting at 10 pixels from the top-left of the legend area
    let vertical_spacing = 20;       // reduced vertical spacing (in pixels) between items
    let horizontal_gap = 5;         // reduced gap between text and dot (in pixels)
    
    for (i, color) in colors.iter().enumerate() {
        // Calculate the y position for this legend item.
        let y = legend_origin.1 + i as i32 * vertical_spacing;
        // Draw the nurse's label.
        legend_area.draw(&Text::new(
            format!("Nurse {}", i + 1),
            (legend_origin.0, y),
            ("sans-serif", 22).into_font().color(&BLACK),
        )).unwrap();
        // Assume a fixed text width of about 60 pixels; adjust if needed.
        let dot_x = legend_origin.0 + 120 + horizontal_gap;
        // Draw a small colored dot next to the text.
        legend_area.draw(&Circle::new(
            (dot_x, y+5),
            5, // radius for the dot
            color.filled(),
        )).unwrap();
    }

    root.present().unwrap();
    println!("Solution diagram saved as {}", output_path);
}

/// Converts a color from HSL (Hue, Saturation, Lightness) to RGB (Red, Green, Blue).
/// 
/// # Parameters:
/// - `h`: The hue angle in degrees.
/// - `s`: The saturation component (0.0 to 1.0).
/// - `l`: The lightness component (0.0 to 1.0).
/// 
/// # Returns:
/// A tuple `(r, g, b)` where each component is an 8-bit unsigned integer representing the corresponding RGB value.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> (u8, u8, u8) {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;
    let (r, g, b) = match h as u32 {
        0..=59   => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179=> (0.0, c, x),
        180..=239=> (0.0, x, c),
        240..=299=> (x, 0.0, c),
        _        => (c, 0.0, x),
    };
    (
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// Draws a small arrow at the end of a route segment to indicate direction.
/// 
/// This function draws two short line segments forming an arrowhead at the specified end point of a route segment.
/// The arrow is oriented based on the provided angle, and its size is defined internally.
/// 
/// # Parameters:
/// - `chart`: A mutable reference to the chart context used for drawing.
/// - `end`: A tuple `(f64, f64)` representing the end point of the route segment.
/// - `angle`: The angle (in degrees) of the route segment, used to orient the arrow.
/// - `color`: A reference to an `RGBColor` specifying the color of the arrow.
fn draw_arrow(
    chart: &mut ChartContext<BitMapBackend, Cartesian2d<RangedCoordf64, RangedCoordf64>>,
    end: (f64, f64),
    angle: f64,
    color: &RGBColor,
) {
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
