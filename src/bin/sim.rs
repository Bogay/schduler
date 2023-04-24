use itertools::{Itertools, MinMaxResult};
use rand::prelude::*;
use rayon::prelude::*;
use std::{
    cmp::Reverse,
    collections::{HashMap, HashSet},
};

#[derive(Debug, Clone)]
struct Room {
    name: String,
    cap: usize,
}

#[derive(Debug, Default, Clone)]
struct Section {
    staffs: Vec<Vec<usize>>,
}

fn gen_rooms(r: usize, n: usize) -> Vec<Room> {
    let cap = (n as f64 / r as f64).ceil() as usize;
    (0..r)
        .map(|i| Room {
            name: format!("Room {i}"),
            cap,
        })
        .collect()
}

#[allow(dead_code)]
fn solve_v1(mut rooms: Vec<Room>, t: usize, n: usize) -> Vec<Section> {
    rooms.sort_by_key(|r| Reverse(r.cap));
    let cap_set: HashSet<_> = rooms.iter().map(|r| r.cap).collect();
    let mut combs: HashMap<_, _> = cap_set
        .iter()
        .map(|c| (c, (0..n).combinations(*c).collect::<Vec<_>>()))
        .collect();
    let mut secs = vec![];
    let mut weights: Vec<_> = (0..n).into_iter().map(|_| vec![0usize; n]).collect();
    let mut assigned_room = vec![HashMap::new(); n];
    let rng = &mut thread_rng();

    for _ in 0..t {
        let mut used = HashSet::<usize>::new();
        let mut sec = Section::default();

        for (room_id, room) in rooms.iter().enumerate() {
            let comb = combs.get_mut(&room.cap).expect("failed to get comb");

            let key_fn = |c: &Vec<usize>| {
                let paired_penalty = c
                    .iter()
                    .combinations(2)
                    .map(|v| weights[*v[0]][*v[1]])
                    .sum::<usize>();
                let reused_penalty = c
                    .iter()
                    .map(|x| (used.contains(x) as usize) * 100000000)
                    .sum::<usize>();
                let reassigned_penalty = c
                    .iter()
                    .map(|x| assigned_room[*x].get(&room_id).unwrap_or(&0) * 10000000)
                    .sum::<usize>();
                let same_cap_rooms = rooms
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (r.cap == room.cap).then_some(i))
                    .collect::<HashSet<_>>();
                let reassigned_same_cap_penalty = c
                    .iter()
                    .flat_map(|x| {
                        assigned_room[*x]
                            .iter()
                            .map(|(r, c)| c * (same_cap_rooms.contains(r) as usize) * 10000000)
                    })
                    .sum::<usize>();

                let penalty = [
                    paired_penalty,
                    reused_penalty,
                    reassigned_penalty,
                    reassigned_same_cap_penalty,
                ]
                .into_iter()
                .sum::<usize>();
                // eprintln!("Key({c:?}): {penalty:?}");
                penalty
            };
            comb.sort_by_cached_key(key_fn);
            // eprintln!("Sorted: {:?}", comb);
            let filtered_comb = comb
                .iter()
                .filter_map(|c| (key_fn(c) < 1000000).then_some(c))
                .collect::<Vec<_>>();
            let c = *filtered_comb.choose(rng).unwrap_or(&&comb[0]);
            for v in c.iter().combinations(2) {
                let (&&a, &&b) = v
                    .iter()
                    .collect_tuple()
                    .expect("combinations(2) should return 2-tuple");
                weights[a][b] += 1;
                weights[b][a] += 1;
            }

            // no one is used before
            assert!(!c.iter().any(|x| used.contains(x)));
            used.extend(c);

            for x in c {
                *assigned_room[*x].entry(room_id).or_default() += 1;
            }

            sec.staffs.push(c.clone());
        }

        secs.push(sec);
    }

    secs
}

fn solve_v2(rooms: &mut Vec<Room>, t: usize, n: usize) -> Vec<Section> {
    let mut secs = vec![];
    let mut weights = vec![vec![0; n]; n];
    let mut acc = vec![0; rooms.len()];
    let mut assigned_room = vec![HashMap::new(); n];
    let rng = &mut thread_rng();

    for _ in 0..t {
        let mut used = HashSet::<usize>::new();
        let mut sec = Section::default();
        sec.staffs.resize(rooms.len(), vec![]);

        // determine where to assign staff i
        let mut staffs = (0..n).collect_vec();
        staffs.shuffle(rng);
        for i in staffs {
            let weight_fn = |i: usize, c: &Vec<usize>, room: &Room, room_id: usize| {
                let paired_penalty = c.iter().map(|j| weights[i][*j]).sum::<usize>();
                const REASSIGN_BONUS: usize = 10000000;
                let reassigned_penalty =
                    assigned_room[i].get(&room_id).unwrap_or(&0) * REASSIGN_BONUS * 10;
                let same_cap_rooms = rooms
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (r.cap == room.cap).then_some(i))
                    .collect::<HashSet<_>>();
                let reassigned_same_cap_penalty = assigned_room[i]
                    .iter()
                    .map(|(r, c)| c * (same_cap_rooms.contains(r) as usize) * REASSIGN_BONUS)
                    .sum::<usize>();
                let cap_penalty = REASSIGN_BONUS * 100 * c.len();
                let acc_penalty = acc[room_id] * REASSIGN_BONUS
                    + c.iter().map(|x| *x * REASSIGN_BONUS).sum::<usize>();

                let penalty = [
                    paired_penalty,
                    cap_penalty,
                    reassigned_penalty,
                    reassigned_same_cap_penalty,
                    acc_penalty,
                ]
                .into_iter()
                .sum::<usize>();
                // eprintln!("Key({c:?}): {penalty:?}");
                penalty
            };

            let picked_room_id = rooms
                .iter()
                .enumerate()
                .filter(|(room_id, room)| sec.staffs[*room_id].len() < room.cap)
                .min_set_by_key(|(room_id, room)| {
                    weight_fn(i, &sec.staffs[*room_id], room, *room_id)
                })
                .first()
                .expect("there should be at least one room")
                .0;

            for j in &sec.staffs[picked_room_id] {
                weights[i][*j] += 1;
                weights[*j][i] += 1;
            }
            used.insert(i);
            *assigned_room[i].entry(picked_room_id).or_default() += 1;
            sec.staffs[picked_room_id].push(i);
            acc[picked_room_id] += 1;
        }

        secs.push(sec);
    }

    secs
}

fn calculate_weights(result: &Vec<Section>) -> Vec<Vec<usize>> {
    let n = result[0].staffs.iter().map(|s| s.len()).sum();
    let mut weights: Vec<_> = (0..n).into_iter().map(|_| vec![0usize; n]).collect();

    for sec in result {
        for s in &sec.staffs {
            for v in s.iter().combinations(2) {
                let (&&a, &&b) = v
                    .iter()
                    .collect_tuple()
                    .expect("combinations(2) should return 2-tuple");
                weights[a][b] += 1;
                weights[b][a] += 1;
            }
        }
    }

    weights
}

fn inspect_result(result: &Vec<Section>) {
    let n = result[0].staffs.iter().map(|s| s.len()).sum();
    let weights = calculate_weights(result);

    eprintln!("Weights:");
    for row in &weights {
        eprintln!("{row:?}");
    }
    eprintln!("Staff assigned to rooms:");
    for s in 0..n {
        let mut count = vec![0; result[0].staffs.len()];
        for sec in result {
            for (room_id, room) in sec.staffs.iter().enumerate() {
                count[room_id] += room.contains(&s) as usize;
            }
        }
        eprintln!("{s:>2}: {count:?}");
    }
}

#[allow(dead_code)]
fn count_tuples(result: &Vec<Section>) -> HashMap<Vec<usize>, usize> {
    let n = result[0].staffs.iter().map(|s| s.len()).sum();
    let mut counter = HashMap::<Vec<usize>, usize>::new();

    for sec in result {
        {
            // one staff should be assigned to only one room in each section
            let mut counter = vec![0; n];
            for c in sec.staffs.iter().flatten() {
                assert_eq!(counter[*c], 0);
                counter[*c] = 1;
            }
        }
        for s in &sec.staffs {
            assert!(!s.is_empty());
            if s.len() == 1 {
                continue;
            }

            if let Some(c) = counter.get_mut(s) {
                *c += 1;
            } else {
                counter.insert(s.clone(), 1);
            }
        }
    }
    counter
}

fn energy(results: &Vec<Section>) -> usize {
    let mut acc = 0;
    let n = results[0].staffs.iter().map(|s| s.len()).sum();
    let len = results[0].staffs.len();
    let mut count = vec![0; len];
    for s in 0..n {
        for sec in results {
            for (room, cnt) in sec.staffs.iter().zip(&mut count) {
                *cnt += room.contains(&s) as usize;
            }
        }
        acc += count.iter().map(|x| x.pow(2)).sum::<usize>();
        count.fill(0);
    }

    acc
}

fn energy2(results: &Vec<Section>, rooms: &Vec<Room>) -> usize {
    let mut acc = energy(results);
    for sec in results {
        let MinMaxResult::MinMax(mn, mx) = sec.staffs
            .iter()
            .zip(rooms)
            .map(|(s, r)| r.cap - s.len())
            .minmax() else { panic!("there should be at least 2 rooms"); };
        acc += mx * (mx - mn);
    }

    acc
}

fn neighbor(mut results: Vec<Section>, rooms: &Vec<Room>, rng: &mut ThreadRng) -> Vec<Section> {
    for _ in 0..rng.gen_range(1..=3) {
        let sec = results
            .choose_mut(rng)
            .expect("results should not be empty");

        let (a, b) = (0..sec.staffs.len())
            .collect_vec()
            .choose_multiple(rng, 2)
            .cloned()
            .collect_tuple()
            .expect("failed to choose 2 indexes from section");

        // let a = rng.gen_range(0..sec.staffs.len());
        // let b = rng.gen_range(0..sec.staffs.len());

        let l = sec.staffs[a].pop().unwrap();
        let r = sec.staffs[b].pop().unwrap();
        sec.staffs[a] = [r].into_iter().chain(sec.staffs[a].drain(..)).collect_vec();
        sec.staffs[b] = [l].into_iter().chain(sec.staffs[b].drain(..)).collect_vec();
        // sec.staffs[a].push(r);
        // sec.staffs[b].push(l);

        // if sec.staffs[a].len() == 1 && sec.staffs[b].len() > 1 {
        //     let x = sec.staffs[b].pop().unwrap();
        //     sec.staffs[a].push(x);
        // } else if sec.staffs[b].len() == 1 && sec.staffs[a].len() > 1 {
        //     let x = sec.staffs[a].pop().unwrap();
        //     sec.staffs[b].push(x);
        // }

        // if rng.gen() && sec.staffs[a].len() < rooms[a].cap && sec.staffs[b].len() > 1 {
        //     let x = sec.staffs[b].pop().unwrap();
        //     sec.staffs[a].push(x);
        // } else if rng.gen() && sec.staffs[b].len() < rooms[b].cap && sec.staffs[a].len() > 1 {
        //     let x = sec.staffs[a].pop().unwrap();
        //     sec.staffs[b].push(x);
        // }
    }

    results
}

fn simulated_annealing(mut results: Vec<Section>, rooms: &Vec<Room>) -> Vec<Section> {
    let mut temperature = 5000000.;
    const COOLING_RATE: f64 = 0.00002;
    const COOLING_MUL: f64 = 1. - COOLING_RATE;
    let mut rng = thread_rng();
    let mut e = energy2(&results, &rooms);

    while temperature > 1. {
        let nxt = neighbor(results.clone(), &rooms, &mut rng);
        let ep = energy2(&nxt, &rooms);
        if ep < e || rng.gen_bool((-((ep - e) as f64) / temperature).exp()) {
            results = nxt;
            e = ep;
        }

        temperature *= COOLING_MUL;
    }

    results
}

fn main() {
    let n = 13;
    let r = 6;
    let mut rooms = gen_rooms(r, n);
    // rooms[0].cap = 4;
    // rooms[1].cap = 4;
    let t = 6;
    let sample_count = 128;

    let results = (0..sample_count)
        .into_par_iter()
        .map(|_| {
            let mut rooms = rooms.clone();
            let result = solve_v2(&mut rooms, t, n);
            // inspect_result(&result);
            // eprintln!("before: {}", energy(&result));
            let result = simulated_annealing(result, &rooms);
            // inspect_result(&result);
            // eprintln!("after: {}", energy(&result));

            result
        })
        .collect::<Vec<_>>();

    // for result in &results {
    //     eprintln!("e: {}", energy(result));
    // }

    let result = results
        .into_iter()
        .min_by_key(energy)
        .expect("failed to find best result");
    inspect_result(&result);
    eprintln!("fin e: {}", energy(&result));

    let num_w = n.to_string().len() + 1;
    let max_n = result
        .iter()
        .flat_map(|r| r.staffs.iter().map(|x| x.len()))
        .max()
        .unwrap();
    let str_w = (max_n - 1) * 2 + max_n * num_w;
    for sec in &result {
        for s in &sec.staffs {
            let s = s.iter().map(|x| format!("{x:num_w$}")).join(", ");
            print!("[{s:>str_w$}] ");
        }

        println!("");
    }
}
